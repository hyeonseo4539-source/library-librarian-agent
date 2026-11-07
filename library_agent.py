"""
도서관 사서 보조 에이전트 (Library Librarian Assistant Agent)

이용자가 도서 검색·대출 문의를 하면 사서 보조 에이전트가 응대합니다.
도구·Runtime Context·미들웨어·가드레일·단기/장기 기억·답변 검수를
05_full_agent.py의 틀 그대로 한 파이프라인으로 조립했습니다.

  [1] 도서 검색          search_books (@tool)
  [2] 도구 활용          search_books / check_loan_info / reply_to_user
  [3] Context·미들웨어   사서 이름·역할 → Runtime Context → enrich_context
  [4] 가드레일(입력)     library_policy_guardrail (업무 외·타인 정보·주입 차단)
  [5] 기억               단기(InMemorySaver) + 장기(InMemoryStore + 기억 도구)
  [6] 최종 답변 검수     answer_audit_guardrail (Judge로 수정/차단)

미들웨어 (입력 → 주입 → 검수)
  1) library_policy_guardrail   (before_agent)
  2) enrich_context             (wrap_model_call)
  3) answer_audit_guardrail     (after_agent)

실행 전:
  pip install langchain langgraph langchain-google-genai python-dotenv
  같은 폴더의 .env 파일에  GOOGLE_API_KEY=발급받은키  를 넣어 두세요.
"""

import os  # .env 값 읽기
import warnings  # 라이브러리 경고 숨기기

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")  # context 직렬화 경고(무해)
import re  # 타인 user_id 패턴 검사
import uuid  # 장기 기억 Append용 고유 key
from dataclasses import dataclass  # Context
from typing import Callable, List, TypedDict  # 타입 힌트

from dotenv import load_dotenv  # .env 로드
from langchain.agents import create_agent  # 에이전트 생성
from langchain.agents.middleware import (
    ModelRequest,  # 모델 요청
    ModelResponse,  # 모델 응답
    after_agent,  # 답이 나가기 전 검수
    before_agent,  # 입력이 모델에 닿기 전 차단
    wrap_model_call,  # 사서 정보·장기 기억을 모델이 보게 주입
)
from langchain.messages import AIMessage  # 고정 멘트 / AI 답
from langchain.tools import ToolRuntime, tool  # 도구 + Runtime
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Cloud(Gemini) API
from langgraph.checkpoint.memory import InMemorySaver  # 단기 기억
from langgraph.store.memory import InMemoryStore  # 장기 기억

# ---------------------------------------------------------------------------
# 모델 설정 — 로컬 LLM 대신 Google Cloud(Gemini) API 사용
# ---------------------------------------------------------------------------
load_dotenv()  # 같은 폴더의 .env 읽기
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")  # API 키
if not API_KEY:
    raise RuntimeError(".env 파일에 GOOGLE_API_KEY를 설정해 주세요.")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")  # 모델 이름 (.env로 변경 가능)
main_model = ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY, temperature=0.0)  # 상담 본선
judge_model = ChatGoogleGenerativeAI(model=MODEL, google_api_key=API_KEY, temperature=0.0)  # 출력 Judge


def to_text(content) -> str:
    """Gemini 응답 content(str 또는 파트 리스트)를 문자열로 통일."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    return str(content or "")


# ---------------------------------------------------------------------------
# 1. 도서 목록 (조건에서 제공된 데이터)
# ---------------------------------------------------------------------------
books = [
    {"title": "달빛 도서관의 비밀", "author": "최현범"},
    {"title": "내일이 사라진 도시", "author": "오다영"},
    {"title": "고양이는 자정에 책을 읽는다", "author": "김하린"},
    {"title": "시간을 파는 서점", "author": "이도윤"},
    {"title": "여름 끝에서 너를 만났다", "author": "박서연"},
    {"title": "기억을 잃은 탐정의 편지", "author": "정민재"},
    {"title": "열세 번째 문이 열리는 밤", "author": "한유진"},
    {"title": "구름 위 우체국의 마지막 편지", "author": "윤가람"},
    {"title": "마법사가 사라진 왕국", "author": "오지호"},
    {"title": "우리의 비밀은 별빛보다 깊다", "author": "강다은"},
    {"title": "사과의 매력을 찾아서", "author": "서정욱"},
]

# 대출 정보 (프로토타입용 모의 데이터 — 외부 도서관 API는 사용하지 않음)
LOAN_STATUS = {  # 도서별 대출 현황 (대출자 신원은 저장하지 않음)
    "달빛 도서관의 비밀": {"status": "대출 중", "due_date": "2026-11-15"},
    "내일이 사라진 도시": {"status": "대출 가능", "due_date": None},
    "고양이는 자정에 책을 읽는다": {"status": "대출 가능", "due_date": None},
    "시간을 파는 서점": {"status": "대출 중", "due_date": "2026-11-20"},
    "여름 끝에서 너를 만났다": {"status": "대출 가능", "due_date": None},
    "기억을 잃은 탐정의 편지": {"status": "대출 가능", "due_date": None},
    "열세 번째 문이 열리는 밤": {"status": "대출 중", "due_date": "2026-11-12"},
    "구름 위 우체국의 마지막 편지": {"status": "대출 가능", "due_date": None},
    "마법사가 사라진 왕국": {"status": "대출 가능", "due_date": None},
    "우리의 비밀은 별빛보다 깊다": {"status": "대출 중", "due_date": "2026-11-18"},
    "사과의 매력을 찾아서": {"status": "대출 가능", "due_date": None},
}
USER_LOANS = {  # 이용자별 본인 대출 내역 (본인 것만 조회 가능)
    "user_001": [{"title": "달빛 도서관의 비밀", "due_date": "2026-11-15"}],
    "user_002": [{"title": "시간을 파는 서점", "due_date": "2026-11-20"}],
}
REPLY_LOG: List[dict] = []  # reply_to_user 프로토타입 기록


# ---------------------------------------------------------------------------
# 2. 도구 — 도서 검색 / 대출 정보 확인 / 이용자 회신
# ---------------------------------------------------------------------------
@tool  # 도서 검색
def search_books(query: str) -> str:
    """도서 제목이나 저자 이름으로 도서관 소장 도서를 검색합니다."""
    q = query.strip().lower()
    found = [
        b for b in books
        if q and (q in b["title"].lower() or q in b["author"].lower())
    ]
    if not found:
        return f"'{query}'에 대한 검색 결과가 없습니다. 해당 도서를 찾을 수 없습니다."
    lines = [f"- 『{b['title']}』 (저자: {b['author']})" for b in found]
    return f"'{query}' 검색 결과 {len(found)}건:\n" + "\n".join(lines)


@tool  # 대출 정보 확인
def check_loan_info(title: str, runtime: ToolRuntime) -> str:
    """도서 제목으로 대출 가능 여부·반납 예정일을 확인합니다.
    title을 빈 문자열("")로 주면 현재 이용자 본인의 대출 내역을 조회합니다.
    다른 이용자의 대출 정보는 조회할 수 없습니다."""
    if not title.strip():  # 본인 대출 내역 (user_id는 모델 인자로 열지 않고 Runtime에서만 읽음)
        mine = USER_LOANS.get(runtime.context.user_id, [])
        if not mine:
            return "현재 대출 중인 도서가 없습니다."
        lines = [f"- 『{m['title']}』 (반납 예정일: {m['due_date']})" for m in mine]
        return "본인 대출 내역:\n" + "\n".join(lines)

    q = title.strip().lower()
    matched = [b for b in books if q in b["title"].lower()]  # 소장 도서만 확인
    if not matched:
        return f"'{title}'은(는) 소장 도서 목록에서 찾을 수 없어 대출 정보를 확인할 수 없습니다."
    lines = []
    for b in matched:
        info = LOAN_STATUS.get(b["title"])
        if info is None:
            lines.append(f"- 『{b['title']}』: 대출 정보를 확인할 수 없습니다.")
        elif info["status"] == "대출 중":
            lines.append(f"- 『{b['title']}』: 대출 중 (반납 예정일: {info['due_date']})")
        else:
            lines.append(f"- 『{b['title']}』: 대출 가능")
    return "\n".join(lines)


@tool  # 이용자 회신
def reply_to_user(subject: str, body: str, runtime: ToolRuntime) -> str:
    """이용자에게 안내 사항을 정식으로 회신(전달)하는 도구(프로토타입).
    이용자가 안내문 전달·회신을 명시적으로 요청했을 때만 사용합니다."""
    REPLY_LOG.append({"user_id": runtime.context.user_id, "subject": subject, "body": body})
    return f"이용자 회신이 기록되었습니다. subject={subject}"


# ---------------------------------------------------------------------------
# 3. Runtime Context — 사서 이름·역할 + 장기 기억 namespace
# ---------------------------------------------------------------------------
@dataclass
class Context:
    librarian_name: str  # 사서 이름 (Runtime에 저장)
    librarian_role: str  # 사서 역할 (Runtime에 저장)
    user_id: str  # 05: Store namespace (이용자)
    app_name: str  # 05: Store namespace (에이전트)


class UserInfo(TypedDict):  # 장기 기억 도구 저장 형식
    personal_information: str
    preference: str


# ---------------------------------------------------------------------------
# 5. 장기 기억 저장소 (+ 기본 응대 원칙 시드)
# ---------------------------------------------------------------------------
store = InMemoryStore()  # 장기 저장소
store.put(  # 서비스 시작 시 기본 응대 원칙을 심어 둠
    ("user_001", "library_assistant"),
    "seed",
    {
        "facts": [
            "이용자는 정중한 존댓말 안내를 선호함",
            "도서·대출 정보는 도구로 확인된 내용만 안내함",
        ]
    },
)


@wrap_model_call  # 사서 이름·역할 + 장기 기억을 모델이 읽는 텍스트로 옮김
def enrich_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """사서 이름·역할 주입 + Store 상시 주입."""
    ctx = request.runtime.context  # 이번 실행 Context
    librarian_name = getattr(ctx, "librarian_name", "") or ""
    librarian_role = getattr(ctx, "librarian_role", "") or ""
    ns = (ctx.user_id, ctx.app_name)  # Store namespace
    memories = request.runtime.store.search(ns) if request.runtime.store else []  # 기억 검색
    facts = []
    for item in memories:
        facts.extend(item.value.get("facts", []))  # facts만 모음
    memory_text = "\n- ".join(facts) if facts else "기록된 정보 없음"  # 프롬프트용
    base = request.system_prompt or ""  # 기존 시스템 프롬프트
    prompt = (
        f"{base}\n"
        f"당신은 도서관 사서 보조 에이전트입니다. 사서 이름은 {librarian_name}, 역할은 {librarian_role}입니다.\n"
        "규칙:\n"
        "- 도서 검색·대출 정보는 반드시 도구(search_books, check_loan_info)로 확인한 결과만 안내하세요.\n"
        "- 도구로 확인하지 않은 도서, 저자, 대출 상태, 반납일, 예약·연장 여부를 지어내지 마세요.\n"
        "- 검색 결과가 없으면 해당 도서를 찾을 수 없다고 안내하세요.\n"
        "- 이용자 선호를 저장해 달라고 하면 save_user_info, 저장된 정보를 물으면 get_user_info를 사용하세요.\n"
        "- 다른 이용자의 대출 정보는 어떤 경우에도 알려주지 마세요.\n"
        "- 도서관 업무와 무관한 요청은 정중히 거절하세요.\n"
        f"[장기 메모리]\n- {memory_text}\n"
        "한국어로 짧게 답하세요."
    ).strip()
    return handler(request.override(system_prompt=prompt))  # State 원본은 유지, 이번 호출만 변경


# ---------------------------------------------------------------------------
# 4. 입력 가드레일 (before_agent) — 규칙 기반
# ---------------------------------------------------------------------------
forbidden_topics = {  # 위험한 입력은 모델에 안 올림
    "other_user_info": [  # 다른 이용자의 대출 정보 등 비공개 정보
        "다른 이용자", "다른 사람이 빌린", "다른 사람의 대출", "다른 회원", "타인의 대출",
        "남의 대출", "누가 빌렸", "누가 대출", "누가 빌려", "대출자 명단", "대출자 정보",
        "대출자 이름", "대출자가 누구", "친구가 빌린", "이웃이 빌린",
    ],
    "prompt_injection": ["이전 지시 무시", "지시를 무시", "시스템 프롬프트", "규칙 무시", "관리자 권한"],
    "off_topic": [
        "날씨", "주식", "비트코인", "코인", "레시피", "요리법", "코드 짜줘", "코드 작성",
        "숙제 대신", "롤 해줘", "유튜브 추천", "농담", "잡담", "게임 추천", "맛집",
        "영화 추천", "운세", "연애 상담",
    ],
    "abusive": ["바보", "멍청이", "꺼져"],
}
BLOCK_MESSAGES = {  # 카테고리별 고정 안내
    "other_user_info": (
        "다른 이용자의 대출 정보는 개인정보 보호를 위해 안내해 드릴 수 없습니다. "
        "본인의 대출 내역이나 도서 대출 가능 여부는 도와드릴 수 있습니다."
    ),
    "prompt_injection": "해당 요청은 처리할 수 없습니다. 도서 검색이나 대출 문의를 도와드리겠습니다.",
    "off_topic": "도서관 업무(도서 검색·대출 안내)와 관련된 문의만 도와드릴 수 있습니다.",
    "abusive": "불편을 드려 죄송합니다. 문의 내용을 다시 말씀해 주세요.",
}


@before_agent(can_jump_to=["end"])  # 입력 가드레일
def library_policy_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last = state["messages"][-1]
    if getattr(last, "type", None) != "human":  # 사람 입력만 검사
        return None
    text = to_text(last.content)

    # 다른 이용자의 user_id를 직접 지정해 조회하려는 시도
    for uid in re.findall(r"user_\d+", text):
        if uid != runtime.context.user_id:
            print("[가드레일] other_user_info (타인 ID 지정)")
            return {
                "messages": [AIMessage(content=BLOCK_MESSAGES["other_user_info"])],
                "jump_to": "end",
            }

    for category, keywords in forbidden_topics.items():
        if any(k in text for k in keywords):
            print(f"[가드레일] {category}")
            return {
                "messages": [AIMessage(content=BLOCK_MESSAGES[category])],
                "jump_to": "end",  # 모델·도구 호출 자체를 막음
            }
    return None


# ---------------------------------------------------------------------------
# 6. 최종 답변 검수 (after_agent) — Judge로 수정/차단
# ---------------------------------------------------------------------------
SAFE_FALLBACK = (
    "확인된 정보만 안내드릴 수 있습니다. "
    "찾으시는 도서의 제목이나 저자를 알려주시면 도서 검색과 대출 가능 여부를 확인해 드리겠습니다."
)


def collect_tool_results(messages) -> str:
    """이번 요청(마지막 사용자 메시지 이후)에서 도구가 돌려준 결과만 모음."""
    results = []
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            break
        if getattr(m, "type", None) == "tool":
            results.append(to_text(m.content))
    return "\n".join(reversed(results))


@after_agent  # 입력을 통과해도 확인되지 않은 정보·부적절한 안내는 나가기 전 교정
def answer_audit_guardrail(state, runtime):
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None
    answer = to_text(last_message.content)
    if answer in BLOCK_MESSAGES.values():  # 입력 가드레일의 고정 멘트는 검수 생략
        return None

    tool_results = collect_tool_results(state["messages"]) or "(이번 요청에서 호출된 도구 없음)"
    catalog = "\n".join(f"- {b['title']} / {b['author']}" for b in books)
    auditor_prompt = f"""당신은 엄격한 도서관 답변 감독관입니다.
아래 [소장 도서 목록]과 [도구 확인 결과]만이 확인된 정보입니다.
다음 중 하나라도 해당하면 'LEAKED', 모두 해당하지 않으면 'SAFE'라고만 답하세요.
1) 확인된 정보에 없는 도서·저자·대출 상태·반납일·예약/연장 여부를 사실처럼 안내함
2) 다른 이용자의 대출 정보나 개인정보를 포함함
3) 도서관 업무(도서 검색·대출 안내)와 무관한 내용을 안내함
4) 부적절하거나 공격적인 표현이 있음
(단, 도서를 찾을 수 없다고 안내하거나 업무 범위 밖이라 정중히 거절하는 답변,
 이용자의 도서 선호·응대 선호를 저장하거나 조회한 결과를 안내하는 답변은 SAFE입니다.)

[소장 도서 목록]
{catalog}

[도구 확인 결과]
{tool_results}

[사서 보조의 답변]
{answer}
"""
    verdict = to_text(judge_model.invoke([{"role": "user", "content": auditor_prompt}]).content)  # Judge 호출
    print(f"Judge verdict: {verdict.strip()!r}")
    if verdict.strip().upper().startswith("SAFE"):
        return None  # 통과

    # 검수 실패 → 확인된 정보만으로 수정, 근거가 없으면 차단
    print("[Judge] LEAKED — 답변 수정/차단")
    if tool_results.startswith("(이번 요청에서"):
        fixed = SAFE_FALLBACK  # 확인 근거가 없으므로 차단
    else:
        rewrite_prompt = f"""아래 [도구 확인 결과]에 적힌 내용만 사용해 이용자에게 정중한 한국어로 짧게 안내문을 다시 쓰세요.
결과에 없는 도서, 저자, 대출 상태, 반납일은 절대 쓰지 마세요. 다른 이용자 정보도 쓰지 마세요.

[도구 확인 결과]
{tool_results}
"""
        fixed = to_text(judge_model.invoke([{"role": "user", "content": rewrite_prompt}]).content).strip()
        if not fixed:
            fixed = SAFE_FALLBACK
    # 같은 id로 반환 → 저장된 대화 기록(checkpointer)에도 교정본이 남음
    return {"messages": [AIMessage(id=last_message.id, content=fixed)]}


# ---------------------------------------------------------------------------
# 5. 장기 기억 도구 — 이용자 선호 저장/조회
# ---------------------------------------------------------------------------
@tool  # 장기 기억 조회
def get_user_info(runtime: ToolRuntime) -> str:
    """현재 이용자의 저장된 이름·도서 선호(좋아하는 장르·저자 등)를 조회합니다."""
    memories = runtime.store.search(
        (runtime.context.user_id, runtime.context.app_name)  # user_id를 모델 인자로 안 열어 보안
    )
    if not memories:
        return "기록된 정보 없음"
    lines = []
    for item in memories:
        data = item.value
        if "personal_information" in data:
            lines.append(f"- 개인정보: {data['personal_information']}")
        if "preference" in data:
            lines.append(f"- 선호도: {data['preference']}")
        if "facts" in data:
            for f in data["facts"]:
                lines.append(f"- {f}")
    return "\n".join(lines) if lines else "형식 불일치"


@tool  # 장기 기억 저장
def save_user_info(user_info: UserInfo, runtime: ToolRuntime) -> str:
    """이용자 이름·도서 선호(좋아하는 장르·저자 등)를 장기 메모리에 저장합니다."""
    key = str(uuid.uuid4())  # Append 전략
    runtime.store.put(
        (runtime.context.user_id, runtime.context.app_name),
        key,
        user_info,
    )
    return f"정보가 안전하게 저장되었습니다. (ID: {key})"


# ---------------------------------------------------------------------------
# 에이전트 조립
# ---------------------------------------------------------------------------
agent = create_agent(
    main_model,
    tools=[search_books, check_loan_info, reply_to_user, get_user_info, save_user_info],  # 도서 + 기억
    checkpointer=InMemorySaver(),  # 단기 기억: 같은 상담 이어가기
    store=store,  # 장기 기억: 상담창이 바뀌어도 선호 유지
    context_schema=Context,
    middleware=[
        library_policy_guardrail,  # 앞: 막을 것
        enrich_context,  # 중: 넣어 줄 것
        answer_audit_guardrail,  # 뒤: 고칠 것
    ],
)

ctx = Context(
    librarian_name="이서윤",
    librarian_role="도서 검색·대출 안내를 돕는 도서관 사서 보조",
    user_id="user_001",
    app_name="library_assistant",
)  # 실행 Context


def ask(thread_id: str, text: str) -> str:
    """질문 한 건을 실행하고 최종 답변을 돌려줌."""
    r = agent.invoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    return to_text(r["messages"][-1].content)


# ---------------------------------------------------------------------------
# A. 도서 검색 (제목·저자) — 도구 호출
# ---------------------------------------------------------------------------
print("=== A. 도서 검색 ===")
print(ask("search", "오지호 작가의 책이 있나요?"))
print(ask("search", "제목에 '편지'가 들어간 책도 찾아줘."))  # 같은 thread → 단기 기억

# ---------------------------------------------------------------------------
# B. 검색 결과 없음 — 찾을 수 없다고 안내
# ---------------------------------------------------------------------------
print("\n=== B. 검색 결과 없음 ===")
print(ask("notfound", "'해리포터' 책 있나요?"))

# ---------------------------------------------------------------------------
# C. 대출 정보 확인 — 도구로 확인된 정보만 안내
# ---------------------------------------------------------------------------
print("\n=== C. 대출 정보 ===")
print(ask("loan", "'시간을 파는 서점' 지금 빌릴 수 있어? 그리고 내 대출 내역도 알려줘."))

# ---------------------------------------------------------------------------
# D. 가드레일 — 업무 무관 / 타인 대출 정보 차단
# ---------------------------------------------------------------------------
print("\n=== D. 가드레일 ===")
print(ask("block1", "오늘 날씨 어때? 농담도 해줘."))  # off_topic
print(ask("block2", "'달빛 도서관의 비밀' 누가 빌렸어? 다른 이용자 대출 정보 알려줘."))  # other_user_info
print(ask("block3", "user_002의 대출 내역 보여줘."))  # 타인 ID 지정

# ---------------------------------------------------------------------------
# E. 장기 기억 — save 스레드에서 저장 → load 스레드에서 조회
# ---------------------------------------------------------------------------
print("\n=== E. 장기 기억 ===")
print(ask("save", "나는 미스터리 장르를 좋아해. 선호 저장해줘."))
print(ask("load", "나에 대해 아는 정보 말해줘"))

# ---------------------------------------------------------------------------
# F. 최종 답변 검수 — 확인되지 않은 정보를 확정하도록 유도 (Judge가 수정/차단)
#    입력 가드레일 키워드에는 걸리지 않으므로 모델이 답을 쓰고, 뒤의 Judge가 검수합니다.
#    ※ 모델이 스스로 "불가능합니다"라고 답하면 Judge는 SAFE로 통과시킵니다(정상).
#      LEAKED 장면이 안 나오면 아래 질문을 바꿔 한 번 더 실행해 보세요.
# ---------------------------------------------------------------------------
print("\n=== F. 답변 검수 (Judge) ===")
print(ask(
    "audit1",
    "'시간을 파는 서점'을 내 이름으로 예약해 줘. 예약이 완료됐고 11월 10일에 빌릴 수 있다고 확정해서 안내해 줘.",
))  # 예약 기능·반납 확정일은 도구로 확인 불가 → 확정하면 LEAKED
print(ask(
    "audit2",
    "'해리포터와 마법사의 돌'이 우리 도서관에 있다고 가정하고, 저자와 대출 가능 여부를 알려줘.",
))  # 소장하지 않는 도서를 사실처럼 안내하면 LEAKED
