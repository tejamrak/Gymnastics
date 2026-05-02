import streamlit as st
import base64, os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

def _logo_b64():
    path = os.path.join(os.path.dirname(__file__), "Logo.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""
LOGO_B64 = _logo_b64()

def _img_data(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        return ""
    ext = filename.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "avif": "image/avif"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

DISC_IMGS = {
    "artistic":   _img_data("artistic woman.jpg"),
    "rhythmic":   _img_data("Rhythmic-gymnastics.jpg"),
    "aerobic":    _img_data("aerobic.jpg"),
    "trampoline": _img_data("trampoline.jpg"),
    "tumbling":   _img_data("tumbling.webp"),
    "acrobatic":  _img_data("acrobatic.jpg"),
    "parkour":    _img_data("parkour.jpg"),
}

BACKGROUND_IMG = _img_data("background.jpg")
GRED_IMG = _img_data("gred.avif")
BG_IMGS = [
    _img_data("background.jpg"),
    _img_data("background 2.jpg"),
    _img_data("background 3.webp"),
    _img_data("background 4.jpg"),
]

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gymnastics Portal | Gimnastika",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — FIG-inspired colour scheme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background-color: #FAFAFA;
}
.main .block-container {
    padding: 2.8rem 3.5rem 4rem 3.5rem;
    max-width: 1140px;
}

/* ── Sidebar collapse button — fades into background ── */
[data-testid="stSidebarCollapseButton"] button {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    opacity: 0.25;
    transition: opacity 0.3s ease;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    opacity: 1;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: #C4B5FD !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C0219 0%, #1A0838 55%, #12062A 100%);
    border-right: 1px solid #2D1459;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #D8CCEE !important;
}
[data-testid="stSidebar"] .stImage {
    display: flex;
    justify-content: center;
    margin: 18px auto 8px auto;
}
[data-testid="stSidebar"] .stImage img {
    display: block;
    margin: 0 auto;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] button {
    background-color: transparent;
    color: #D8CCEE !important;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0 8px 8px 0;
    text-align: center;
    font-size: 16px !important;
    font-weight: 500;
    padding: 10px 16px;
    margin: 2px 0;
    width: 100%;
    letter-spacing: 0.3px;
}
[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] button span,
[data-testid="stSidebar"] .stButton > button p {
    font-size: 16px !important;
    color: #D8CCEE !important;
    line-height: normal !important;
    text-align: center !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(124, 58, 237, 0.22);
    border-left-color: #A78BFA;
    color: white !important;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0C0219 0%, #4C1D95 48%, #0F766E 100%);
    color: white;
    padding: 80px 60px;
    border-radius: 4px;
    margin-bottom: 44px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -90px; right: -70px;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: radial-gradient(circle, #C084FC 0%, transparent 65%);
    opacity: 0.2;
}
.hero-banner::after {
    content: "";
    position: absolute;
    bottom: -110px; left: -60px;
    width: 400px; height: 400px;
    border-radius: 50%;
    background: radial-gradient(circle, #2DD4BF 0%, transparent 65%);
    opacity: 0.16;
}
.hero-banner h1 {
    color: white !important;
    font-size: 3.4em;
    font-weight: 800;
    margin-bottom: 16px;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
    text-transform: uppercase;
}
.hero-banner p {
    font-size: 1.25em;
    opacity: 0.88;
    position: relative;
    z-index: 1;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.75;
}

/* ── Section headers — Metro left-border style ── */
.section-header {
    color: #0C0219;
    padding: 0 0 0 18px;
    margin-bottom: 32px;
    border-left: 6px solid #7C3AED;
    border-bottom: none;
    font-size: 2em;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    line-height: 1.2;
}

/* ── Metro tiles (home page) ── */
.metro-tile {
    padding: 30px 24px 24px 24px;
    margin-bottom: 0;
    border-radius: 2px;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.metro-tile h4 {
    color: white !important;
    font-size: 19px;
    font-weight: 800;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
.metro-tile p {
    color: rgba(255,255,255,0.78) !important;
    font-size: 22px;
    margin: 0;
    line-height: 1.5;
    font-weight: 400;
}

/* ── Result cards (search) ── */
.result-card {
    background: white;
    border-left: 5px solid #7C3AED;
    border-radius: 6px;
    padding: 24px 28px;
    margin: 12px 0;
    border-top: 1px solid #EDE9FE;
    border-right: 1px solid #EDE9FE;
    border-bottom: 1px solid #EDE9FE;
    box-shadow: 0 2px 12px rgba(124,58,237,0.07);
}
.result-card-best {
    background: white;
    border-left: 5px solid #F59E0B;
    border-radius: 6px;
    padding: 24px 28px;
    margin: 12px 0;
    border-top: 1px solid #FDE68A;
    border-right: 1px solid #FDE68A;
    border-bottom: 1px solid #FDE68A;
    box-shadow: 0 4px 18px rgba(245,158,11,0.13);
}
.result-section-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 28px 0 6px 0;
    padding-bottom: 6px;
    border-bottom: 2px solid currentColor;
}
.result-badge {
    display: inline-block;
    background: #F59E0B;
    color: white;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 3px;
    margin-bottom: 10px;
}
.result-title {
    color: #0C0219;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.result-meta {
    color: #9CA3AF;
    font-size: 16px;
    margin-bottom: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.result-content {
    color: #374151;
    line-height: 1.8;
    font-size: 26px;
}

/* ── Search page controls ── */
.stTextInput > div > div > input {
    border: 2px solid #7C3AED !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    font-size: 18px !important;
    background: #FAFAFF !important;
    color: #0C0219 !important;
    box-shadow: 0 2px 8px rgba(124,58,237,0.08) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #5B21B6 !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}
.stSlider [data-testid="stSlider"] > div {
    padding-top: 6px;
}
.stSlider .rc-slider-track {
    background: #7C3AED !important;
}
.stSlider .rc-slider-handle {
    border-color: #7C3AED !important;
    background: #7C3AED !important;
}
.stSelectbox > div > div {
    border: 2px solid #7C3AED !important;
    border-radius: 6px !important;
    background: #FAFAFF !important;
}

/* ── General buttons ── */
.stButton > button {
    background: #7C3AED;
    color: white !important;
    border: none;
    border-radius: 2px;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: none;
    transition: background 0.18s ease;
    padding: 12px 22px;
}
.stButton > button:hover {
    background: #0D9488;
    box-shadow: none;
}

/* ── Typography ── */
h1, h2, h3, h4 {
    color: #0C0219;
    font-weight: 700;
}
.stMarkdown p, .stMarkdown li,
.element-container p, .element-container li,
p, li {
    color: #2D2D2D;
    line-height: 1.9;
    font-size: 30px !important;
    text-align: justify !important;
}
/* Sidebar button text — declared after global p rule so order+specificity both win */
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {
    font-size: 16px !important;
    text-align: center !important;
    color: #D8CCEE !important;
    line-height: normal !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary div {
    font-size: 16px !important;
    text-align: center !important;
    color: #D8CCEE !important;
    line-height: normal !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    font-size: 16px !important;
}
strong {
    color: #0C0219;
    font-weight: 700;
}
hr {
    border-color: #E5E7EB;
    margin: 32px 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #EDE9FE;
}
.stTabs [data-baseweb="tab"] {
    font-size: 26px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 13px 22px;
    border-radius: 0;
    color: #6B7280;
}
.stTabs [aria-selected="true"] {
    color: #7C3AED !important;
    border-bottom: 3px solid #7C3AED;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #E5E7EB;
    border-radius: 2px;
    margin-bottom: 10px;
    background: white;
}
[data-testid="stExpander"] summary {
    font-size: 26px;
    font-weight: 600;
    color: #0C0219;
    padding: 16px 20px;
}

/* ── Sidebar expander (disciplines dropdown) ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0;
    margin: 0;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #D8CCEE !important;
    font-size: 16px !important;
    font-weight: 500;
    padding: 10px 16px;
    background: transparent;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    display: flex !important;
    justify-content: center !important;
    align-items: center;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary span,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    font-size: 16px !important;
    text-align: center !important;
    flex: 1;
}
[data-testid="stSidebar"] [data-testid="stExpander"]:has(details[open]) {
    border-left-color: #A78BFA;
    background: rgba(124, 58, 237, 0.1);
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] .stButton > button {
    font-size: 15px;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
    padding: 7px 14px 7px 32px;
    background: transparent;
    box-shadow: none;
    color: #B8A8D8 !important;
    border-left: 3px solid transparent;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] .stButton > button:hover {
    background: rgba(124, 58, 237, 0.18) !important;
    border-left-color: #A78BFA !important;
    color: white !important;
}

/* ── Sidebar logo bottom ── */
.sidebar-logo-bottom {
    text-align: center;
    padding: 32px 0 24px 0;
    margin-top: 20px;
    border-top: 1px solid #ffffff15;
}
.sidebar-logo-bottom img {
    display: block;
    margin: 0 auto;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DOCUMENTS — 19 approved documents across 4 blocks
# ============================================================
DOCUMENTS = [
    {
        "id": 1,
        "title": "What is Gymnastics?",
        "language": "en",
        "category": "Foundations",
        "content": (
            "WHAT IS GYMNASTICS?\n\n"
            "Gymnastics is a sport that combines strength, flexibility, balance, coordination, "
            "and movement technique across a range of disciplines and apparatus.\n\n"
            "Its origins trace back to ancient Greece, where gymnastic exercises were part of "
            "physical and military education. The modern sport took shape in 19th-century Europe, "
            "largely through the work of Friedrich Ludwig Jahn in Germany, widely regarded as the "
            "father of gymnastics. Gymnastics was included in the first modern Olympic Games in "
            "Athens in 1896 and has been part of every Summer Olympics since.\n\n"
            "The sport is governed internationally by the Federation Internationale de Gymnastique "
            "(FIG), founded in 1881 and headquartered in Lausanne, Switzerland. FIG establishes the "
            "competition rules through the Code of Points, organizes the World Championships, and "
            "oversees the Olympic gymnastics program. The organization has member federations in "
            "over 140 countries.\n\n"
            "Athletes typically begin training in early childhood and progress through national "
            "competition systems toward international competition. The highest level of competition "
            "is the Olympic Games."
        ),
    },
    {
        "id": 2,
        "title": "Types of Gymnastics",
        "language": "en",
        "category": "Foundations",
        "content": (
            "TYPES OF GYMNASTICS\n\n"
            "Gymnastics comprises six disciplines, each with its own rules, apparatus, and "
            "physical requirements.\n\n"
            "WOMEN'S ARTISTIC GYMNASTICS (WAG): Female gymnasts compete on four apparatus: "
            "vault, uneven bars, balance beam, and floor exercise.\n\n"
            "MEN'S ARTISTIC GYMNASTICS (MAG): Male gymnasts compete on six apparatus: floor "
            "exercise, pommel horse, still rings, vault, parallel bars, and horizontal bar.\n\n"
            "RHYTHMIC GYMNASTICS: Performed to music using hand apparatus (rope, hoop, ball, "
            "clubs, or ribbon), with an emphasis on flexibility, coordination, and apparatus "
            "technique rather than acrobatic tumbling.\n\n"
            "TRAMPOLINE GYMNASTICS: Athletes perform acrobatic skills on a trampoline, reaching "
            "heights of up to 8 meters. Routines are evaluated on difficulty, execution, and "
            "time of flight.\n\n"
            "AEROBIC GYMNASTICS: Consists of continuous routines performed to music, demonstrating "
            "strength, coordination, and cardiorespiratory endurance. It is competed individually, "
            "in pairs, trios, or groups.\n\n"
            "ACROBATIC GYMNASTICS: Involves groups of 2 to 4 gymnasts performing dynamic and "
            "balance elements together, including throws, catches, and partner balances."
        ),
    },
    {
        "id": 3,
        "title": "Women's Artistic Gymnastics: The 4 Apparatus",
        "language": "en",
        "category": "Foundations",
        "content": (
            "WOMEN'S ARTISTIC GYMNASTICS: THE 4 APPARATUS\n\n"
            "VAULT: The gymnast sprints approximately 25 meters down the runway, takes off from "
            "a springboard, pushes off the vaulting table, and executes a skill in the air before "
            "landing. The flight phase lasts approximately 1 to 2 seconds. Scores are based on "
            "the difficulty of the vault and the quality of execution and landing.\n\n"
            "UNEVEN BARS: Two horizontal bars are set at different heights, with the high bar at "
            "approximately 2.50 m and the low bar at 1.70 m. The gymnast performs a continuous "
            "routine of swings, releases, and catches between the two bars. Evaluation focuses on "
            "swing technique, release skills, and body position.\n\n"
            "BALANCE BEAM: The beam is 5 meters long and 10 centimeters wide, raised 125 "
            "centimeters above the floor. Gymnasts perform a routine of 70 to 90 seconds combining "
            "jumps, turns, acrobatic elements, and choreography. Any step, fall, or loss of balance "
            "results in a deduction.\n\n"
            "FLOOR EXERCISE: The routine is performed on a 12 by 12 meter sprung floor to music, "
            "lasting 70 to 90 seconds. It includes tumbling passes, jumps, turns, and choreographic "
            "sequences. Going outside the boundary lines results in a deduction."
        ),
    },
    {
        "id": 4,
        "title": "Men's Artistic Gymnastics: The 6 Apparatus",
        "language": "en",
        "category": "Foundations",
        "content": (
            "MEN'S ARTISTIC GYMNASTICS: THE 6 APPARATUS\n\n"
            "FLOOR: Tumbling passes and strength elements are performed on a 12 by 12 meter "
            "sprung floor, without musical accompaniment at the international level.\n\n"
            "POMMEL HORSE: The gymnast performs continuous circular and scissor movements on a "
            "leather horse with two handles, supporting the body on the arms throughout. Any stop "
            "in movement or contact with the leg results in a deduction.\n\n"
            "STILL RINGS: Two rings are suspended from cables at 2.75 meters. The gymnast performs "
            "swing elements and static strength holds, such as the iron cross or planche, while "
            "minimizing movement in the rings themselves.\n\n"
            "VAULT: Performed on the same apparatus as in the women's discipline, at higher "
            "difficulty levels.\n\n"
            "PARALLEL BARS: Two bars are set at equal height (195 cm), approximately 42 cm apart. "
            "Gymnasts perform swinging, release, and strength elements both between and above "
            "the bars.\n\n"
            "HORIZONTAL BAR: A single bar is set at 280 cm. Gymnasts perform giant swings and "
            "release skills, briefly releasing the bar entirely, before catching it again and "
            "finishing with a dismount."
        ),
    },
    {
        "id": 5,
        "title": "Rhythmic Gymnastics",
        "language": "en",
        "category": "Foundations",
        "content": (
            "RHYTHMIC GYMNASTICS\n\n"
            "Rhythmic gymnastics combines movement drawn from dance and ballet with the technical "
            "manipulation of hand apparatus. Competitions are held in individual and group formats.\n\n"
            "Gymnasts compete with five types of hand apparatus:\n\n"
            "ROPE: Jumping, throwing, and rotating the rope.\n\n"
            "HOOP: Rolling along the body, throwing, rotating, and passing through.\n\n"
            "BALL: Rolling on the body, bouncing, and throwing.\n\n"
            "CLUBS: Throwing and catching skills performed with two clubs simultaneously.\n\n"
            "RIBBON: Continuous movement that draws shapes in the air; contact with the floor "
            "results in a deduction.\n\n"
            "In the group event, five gymnasts perform together, either with identical apparatus "
            "or a combination of two different types.\n\n"
            "The discipline does not include acrobatic tumbling. Evaluation focuses on flexibility, "
            "body movement technique, and the quality of apparatus handling. Russia, Bulgaria, "
            "Israel, and Spain have historically achieved the strongest international results."
        ),
    },
    {
        "id": 6,
        "title": "Trampoline, Aerobic, and Acrobatic Gymnastics",
        "language": "en",
        "category": "Foundations",
        "content": (
            "TRAMPOLINE, AEROBIC, AND ACROBATIC GYMNASTICS\n\n"
            "TRAMPOLINE GYMNASTICS: Athletes perform a routine of 10 consecutive acrobatic skills "
            "on a trampoline, reaching heights of up to 8 to 10 meters. Each skill must be "
            "different. Routines are evaluated on the difficulty of the skills performed, the "
            "quality of execution, and the total time of flight, which measures how long the "
            "athlete stays in the air across the entire routine. Trampoline also includes two "
            "related disciplines: double mini trampoline (a smaller apparatus with a run-up) and "
            "tumbling (acrobatic skills performed on a sprung runway without a trampoline).\n\n"
            "AEROBIC GYMNASTICS: Routines last 60 to 90 seconds and consist of continuous, "
            "high-intensity movement performed to music. Athletes demonstrate strength, flexibility, "
            "coordination, and cardiorespiratory endurance through a combination of aerobic steps "
            "and gymnastics skills. Competition categories include individual men, individual women, "
            "mixed pairs, trios, and groups of six.\n\n"
            "ACROBATIC GYMNASTICS: Athletes compete in partnerships: women's pairs, men's pairs, "
            "mixed pairs, women's groups (3 athletes), and men's groups (4 athletes). Routines "
            "include two types of elements: balance elements, where partners hold static positions "
            "together, and dynamic elements, where one partner throws, catches, or releases "
            "another. Routines are performed to music and evaluated on difficulty, execution, "
            "and artistry."
        ),
    },
    {
        "id": 7,
        "title": "Types of Gymnastics Competitions",
        "language": "en",
        "category": "Competitions",
        "content": (
            "TYPES OF GYMNASTICS COMPETITIONS\n\n"
            "Gymnastics competitions are organized at several levels, from local club events to "
            "the Olympic Games.\n\n"
            "OLYMPIC GAMES: Held every four years, the Olympics represent the highest level of "
            "gymnastics competition. Countries qualify teams and individual gymnasts based on "
            "results at the preceding World Championships and qualification events.\n\n"
            "WORLD CHAMPIONSHIPS: Organized annually by FIG, the World Championships are the "
            "most important competition outside of the Olympic cycle. Athletes compete in team, "
            "individual all-around, and individual apparatus events.\n\n"
            "WORLD CUP SERIES: A series of individual apparatus competitions held in different "
            "countries throughout the year. Athletes accumulate points toward an overall World "
            "Cup ranking.\n\n"
            "CONTINENTAL CHAMPIONSHIPS: Each continent organizes its own championship. In Europe, "
            "the European Gymnastics Championships are held every two years and are open to all "
            "European FIG member federations.\n\n"
            "At the national level, each country organizes its own competition system. In Slovenia, "
            "the Gymnastics Federation of Slovenia (GZS) oversees national championships, regional "
            "competitions, and age-group events for gymnasts at all levels, from beginners to the "
            "national team."
        ),
    },
    {
        "id": 8,
        "title": "How a WAG Competition Works",
        "language": "en",
        "category": "Competitions",
        "content": (
            "HOW A WAG COMPETITION WORKS\n\n"
            "A Women's Artistic Gymnastics competition typically follows a structured format "
            "across multiple rounds.\n\n"
            "PODIUM TRAINING: Before the competition begins, teams have a period of familiarization "
            "on the competition apparatus, known as podium training. This allows gymnasts to adjust "
            "to the specific equipment and environment.\n\n"
            "QUALIFICATIONS: All teams and individual gymnasts compete in the qualification round. "
            "Each gymnast performs on some or all of the four apparatus. The results from "
            "qualifications determine who advances to the finals and serve as the team competition "
            "scores.\n\n"
            "TEAM FINAL: The top 8 teams from qualifications advance. Each country fields up to "
            "5 gymnasts, with 4 competing on each apparatus and the top 3 scores counting toward "
            "the team total.\n\n"
            "INDIVIDUAL ALL-AROUND FINAL: The top 24 gymnasts from qualifications, with a maximum "
            "of 2 per country, compete on all four apparatus. The combined score across all four "
            "determines the all-around champion.\n\n"
            "EVENT FINALS: The top 8 gymnasts on each individual apparatus from qualifications, "
            "with a maximum of 2 per country, compete again. Each apparatus has its own champion.\n\n"
            "During the competition, gymnasts rotate between apparatus in a fixed order: vault, "
            "uneven bars, balance beam, and floor exercise. Each team or gymnast performs in a "
            "designated subdivision based on qualification scores."
        ),
    },
    {
        "id": 9,
        "title": "What Does It Take to Be a Gymnast?",
        "language": "en",
        "category": "The Athlete",
        "content": (
            "WHAT DOES IT TAKE TO BE A GYMNAST?\n\n"
            "Gymnastics is a sport that requires a combination of physical and mental qualities "
            "developed over many years of training.\n\n"
            "TRAINING VOLUME: Elite gymnasts typically train 7 to 8 hours per day, which amounts "
            "to roughly 40 to 50 hours per week. At younger and lower levels, training is less "
            "intensive, but consistent practice over years is necessary to develop the required "
            "skills.\n\n"
            "STARTING AGE: Most competitive gymnasts begin training between the ages of 3 and 6. "
            "At the elite international level, gymnasts typically reach their peak competitive "
            "years between the ages of 15 and 24 in the women's artistic discipline.\n\n"
            "PHYSICAL QUALITIES: The sport requires flexibility, strength relative to body weight, "
            "coordination, spatial awareness, and the ability to learn and repeat complex movement "
            "patterns precisely.\n\n"
            "MENTAL DEMANDS: Gymnasts must manage performance under pressure, overcome fear when "
            "learning new skills, and maintain consistency across multiple events in a single "
            "competition. Mental preparation is considered as important as physical training at "
            "the elite level.\n\n"
            "EDUCATION: Due to the high number of training hours, elite gymnasts are frequently "
            "homeschooled or follow adapted educational programs. This allows them to maintain "
            "their training schedule while fulfilling their academic requirements."
        ),
    },
    {
        "id": 10,
        "title": "Notable Gymnasts in History",
        "language": "en",
        "category": "The Athlete",
        "content": (
            "NOTABLE GYMNASTS IN HISTORY\n\n"
            "Several gymnasts have shaped the history of Women's Artistic Gymnastics through their "
            "performances and contributions to the sport.\n\n"
            "LARISA LATYNINA (USSR): Competed between 1956 and 1964, winning 18 Olympic medals: "
            "9 gold, 5 silver, and 4 bronze. This remained the record for most Olympic medals by "
            "any athlete for nearly 50 years.\n\n"
            "VERA CASLAVSKA (CZECHOSLOVAKIA): Won 7 Olympic gold medals across the 1964 and 1968 "
            "Games and is regarded as one of the defining figures in the sport's early development.\n\n"
            "OLGA KORBUT (USSR): Gained international attention at the 1972 Munich Olympics, "
            "introducing skills of a difficulty not previously seen in competition and bringing a "
            "new public audience to the sport.\n\n"
            "NADIA COMANECI (ROMANIA): Became the first gymnast in Olympic history to receive a "
            "perfect score of 10.0, at the 1976 Montreal Games at the age of 14. She received "
            "seven perfect scores during those Games.\n\n"
            "SIMONE BILES (USA): Has accumulated more World Championship medals than any other "
            "gymnast in history. She has introduced multiple skills of unprecedented difficulty, "
            "several of which now carry her name in the Code of Points. She competed at the 2016 "
            "and 2020 Olympic Games, winning multiple gold medals at each."
        ),
    },
    {
        "id": 11,
        "title": "How WAG Scoring Works",
        "language": "en",
        "category": "Judging System",
        "content": (
            "HOW WAG SCORING WORKS\n\n"
            "THE OPEN-ENDED SCORING SYSTEM\n\n"
            "Women's Artistic Gymnastics uses an open-ended scoring system introduced after the "
            "2004 Olympic Games. Under this system, scores are not capped at 10.0. Instead, the "
            "final score is the sum of two separate components:\n\n"
            "Final Score = Difficulty Score (D) + Execution Score (E) - Neutral Deductions\n\n"
            "The Difficulty Score reflects what the gymnast attempts, and the Execution Score "
            "reflects how well she performs it. Neutral Deductions are applied for specific rule "
            "violations and are subtracted from the total.\n\n"
            "THE TWO MAIN COMPONENTS\n\n"
            "The Difficulty Score (D) has no maximum limit. A gymnast who includes more difficult "
            "skills and fulfills more composition requirements will receive a higher D score.\n\n"
            "The Execution Score (E) starts at a maximum of 10.0. Judges deduct for technical and "
            "artistic errors throughout the routine. The final E score reflects how cleanly and "
            "precisely the gymnast performed.\n\n"
            "TIEBREAKER RULE\n\n"
            "If two gymnasts finish with the same final score, the gymnast with the higher "
            "Difficulty Score is ranked higher. This rule reflects the principle that greater "
            "technical difficulty is rewarded when execution is equal."
        ),
    },
    {
        "id": 12,
        "title": "The Difficulty Score (D Score)",
        "language": "en",
        "category": "Judging System",
        "content": (
            "THE DIFFICULTY SCORE (D SCORE)\n\n"
            "The Difficulty Score is calculated by the two D-panel judges and consists of three "
            "parts: the value of the gymnast's skills, the Composition Requirements, and any "
            "Connection Value bonuses.\n\n"
            "ELEMENT VALUES\n\n"
            "Every skill in the FIG Code of Points is assigned a difficulty letter from A to I:\n"
            "A = 0.1, B = 0.2, C = 0.3, D = 0.4, E = 0.5, F = 0.6, G = 0.7, H = 0.8, I = 0.9\n\n"
            "On floor, beam, and uneven bars, only the 8 highest-valued elements are counted "
            "toward the D score. On vault, the routine consists of a single skill whose value is "
            "listed directly in the FIG Code of Points.\n\n"
            "COMPOSITION REQUIREMENTS (CR)\n\n"
            "Each apparatus has specific Composition Requirements. Each fulfilled requirement adds "
            "0.5 points to the D score, with a maximum of 2.0 points available from this "
            "component. If a gymnast does not include a required element type, she simply does not "
            "earn those points. There is no separate penalty for the omission.\n\n"
            "CONNECTION VALUE (CV)\n\n"
            "Gymnasts can earn bonus tenths (0.10 or 0.20) by directly connecting certain "
            "elements. These bonuses vary by apparatus and are specified in the FIG Code of Points."
        ),
    },
    {
        "id": 13,
        "title": "The Execution Score (E Score)",
        "language": "en",
        "category": "Judging System",
        "content": (
            "THE EXECUTION SCORE (E SCORE)\n\n"
            "The Execution Score begins at 10.0 for every routine. A panel of five judges "
            "evaluates the gymnast's technical precision, body form, and artistry. Each error "
            "results in a deduction from that starting value.\n\n"
            "To reduce the effect of individual bias, the highest and lowest scores from the five "
            "judges are removed. The final E score is the average of the remaining three.\n\n"
            "THE DEDUCTION SCALE\n\n"
            "The FIG Code of Points uses four standard deduction values:\n"
            "Small error: 0.10\n"
            "Medium error: 0.30\n"
            "Large error: 0.50\n"
            "Fall: 1.00\n\n"
            "A fall is defined as an uncontrolled loss of contact with the apparatus or floor and "
            "always results in a 1.00 deduction.\n\n"
            "MAXIMUM DEDUCTION PER ELEMENT\n\n"
            "For any single skill, the maximum execution deduction is 0.80, regardless of how many "
            "errors are observed in that element. A fall is treated separately and is always "
            "penalized as a 1.00 deduction; it is not subject to this cap.\n\n"
            "EXAMPLES OF EXECUTION ERRORS\n\n"
            "Bent knees: 0.10 or 0.30 depending on severity.\n"
            "Separated legs when they should be together: 0.10 or 0.30.\n"
            "Flexed feet: 0.10 or 0.30.\n"
            "Insufficient height or amplitude of a skill: 0.10 or 0.30.\n"
            "Steps on landing: 0.10 (small step) or 0.30 (large step or stagger).\n"
            "Loss of balance on beam: 0.10, 0.30, or 0.50 depending on severity."
        ),
    },
    {
        "id": 14,
        "title": "The Judging Panel",
        "language": "en",
        "category": "Judging System",
        "content": (
            "THE JUDGING PANEL\n\n"
            "At FIG-sanctioned competitions, each apparatus is evaluated by a panel of judges "
            "divided into two groups with distinct responsibilities.\n\n"
            "D PANEL (DIFFICULTY)\n\n"
            "The D panel consists of two judges (D1 and D2). They identify every skill performed, "
            "verify that Composition Requirements are met, and calculate the total Difficulty "
            "Score. Both judges work independently and then compare results. If their scores "
            "differ beyond the permitted range, the matter is referred to a Superior Jury.\n\n"
            "E PANEL (EXECUTION)\n\n"
            "The E panel consists of five judges (E1 through E5). Each assesses the routine "
            "independently. After the routine, the highest and lowest scores are removed and the "
            "average of the remaining three produces the final E score.\n\n"
            "REFEREE\n\n"
            "The referee oversees the entire process: gives the starting signal, monitors neutral "
            "deductions such as time violations, and ensures the competition follows FIG "
            "regulations.\n\n"
            "INQUIRIES\n\n"
            "After a score is announced, the gymnast's coach may submit a formal inquiry to "
            "challenge the D score only. E score decisions are final and cannot be challenged. "
            "Inquiries must be submitted within the designated time window and require a fee, "
            "which is refunded if the inquiry succeeds."
        ),
    },
    {
        "id": 15,
        "title": "Deductions on Vault and Uneven Bars",
        "language": "en",
        "category": "Judging System",
        "content": (
            "DEDUCTIONS ON VAULT AND UNEVEN BARS\n\n"
            "VAULT\n\n"
            "On vault, each gymnast performs a single skill. Every vault in the FIG Code of "
            "Points has a predetermined difficulty value, so the D score is simply the value of "
            "the vault performed. The E score starts at 10.0 and is assessed by the five E-panel "
            "judges.\n\n"
            "Common deductions on vault:\n"
            "Landing steps: 0.10 (small step) or 0.30 (large step or stagger).\n"
            "Fall on landing: 1.00.\n"
            "Flexed feet during flight: 0.10 or 0.30.\n"
            "Bent knees: 0.10 or 0.30.\n"
            "Low or under-rotated vault: 0.30 or 0.50.\n"
            "Touching the mat with hands on landing: 0.50.\n\n"
            "In vault finals at major competitions, gymnasts perform two vaults and the two scores "
            "are averaged.\n\n"
            "UNEVEN BARS\n\n"
            "A bars routine includes up to 8 counted elements. The gymnast must meet the "
            "Composition Requirements of the apparatus.\n\n"
            "Common deductions on bars:\n"
            "Bent knees or piked position when a straight body is required: 0.10 or 0.30.\n"
            "Flexed feet: 0.10 or 0.30.\n"
            "Insufficient height on release elements: 0.10 or 0.30.\n"
            "Steps or hop on dismount landing: 0.10 or 0.30.\n"
            "Fall from the bar: 1.00.\n\n"
            "FALL RECOVERY ON BARS: When a gymnast falls from the bar, a deduction of 1.00 is "
            "immediately applied to her E score. She then has 30 seconds to remount and continue. "
            "If she does not remount within 30 seconds, the routine is considered terminated and "
            "she receives no credit for any elements she did not perform."
        ),
    },
    {
        "id": 16,
        "title": "Deductions on Balance Beam and Floor Exercise",
        "language": "en",
        "category": "Judging System",
        "content": (
            "DEDUCTIONS ON BALANCE BEAM AND FLOOR EXERCISE\n\n"
            "BALANCE BEAM\n\n"
            "A beam routine must last between 70 and 90 seconds. If the routine exceeds 90 "
            "seconds, a neutral deduction of 0.10 is applied for each full second over the limit.\n\n"
            "Common deductions on beam:\n"
            "Small loss of balance, visible but controlled: 0.10.\n"
            "More significant loss of balance, with a bent or hunched body: 0.30.\n"
            "Loss of balance severe enough to touch the beam with hands: 0.50.\n"
            "Fall from the beam: 1.00.\n"
            "Bent knees: 0.10 or 0.30.\n"
            "Landing step after a skill: 0.10 (small) or 0.30 (large).\n\n"
            "FALL RECOVERY ON BEAM: When a gymnast falls from the beam, a deduction of 1.00 is "
            "immediately applied to her E score. She then has 10 seconds to remount and continue. "
            "If she does not remount within 10 seconds, the routine is terminated and she receives "
            "no credit for any elements she did not perform.\n\n"
            "FLOOR EXERCISE\n\n"
            "A floor routine must last between 70 and 90 seconds and is performed to music. If "
            "the routine exceeds 90 seconds, a neutral deduction of 0.10 per second over the "
            "limit is applied.\n\n"
            "Common deductions on floor:\n"
            "Landing steps: 0.10 (small step) or 0.30 (large step or stagger).\n"
            "Bent knees or legs apart on a salto: 0.10 or 0.30.\n"
            "Flexed feet: 0.10 or 0.30.\n"
            "Out of bounds: 0.30 per foot that lands completely outside the boundary.\n\n"
            "The floor area is 12 by 12 meters. A foot landing fully outside the line is "
            "penalized; a foot on the line is not."
        ),
    },
    {
        "id": 17,
        "title": "Gimnastika v Sloveniji",
        "language": "sl",
        "category": "Slovenija",
        "content": (
            "GIMNASTIKA V SLOVENIJI\n\n"
            "GIMNASTIČNA ZVEZA SLOVENIJE\n\n"
            "Gimnastična zveza Slovenije (GZS) je nacionalna krovska organizacija za gimnastiko "
            "v Sloveniji. Je članica Mednarodne gimnastične zveze (FIG) in Evropske gimnastične "
            "zveze (UEG). GZS organizira tekmovanja na vseh ravneh, od začetniških kategorij do "
            "državnega in mednarodnega nivoja, ter skrbi za razvoj gimnastike v državi.\n\n"
            "SLOVENIJA NA MEDNARODNEM PRIZORIŠČU\n\n"
            "Po osamosvojitvi leta 1991 je Slovenija začela nastopati na mednarodnih tekmovanjih "
            "pod lastno zastavo. Slovenske gimnastičarke so se uvrstile na Olimpijske igre, "
            "Svetovna in Evropska prvenstva ter različna mednarodna tekmovanja serije World Cup. "
            "Kljub temu da je Slovenija majhna država z omejenim številom tekmovalk, je s "
            "posamičnimi nastopi dosegla odmevne rezultate na najvišji ravni.\n\n"
            "RAZVOJ ŠPORTA\n\n"
            "Gimnastika v Sloveniji se začne že v zgodnjem otroštvu. Mladi gimnastičarji se "
            "razvijajo v klubih po vsej državi, ki delujejo v okviru GZS. Naprednejše tekmovalke, "
            "ki dosežejo ustrezen nivo, se vključijo v program državne reprezentance in tekmujejo "
            "po pravilih FIG."
        ),
    },
    {
        "id": 18,
        "title": "Slovenske gimnastičarke",
        "language": "sl",
        "category": "Slovenija",
        "content": (
            "SLOVENSKE GIMNASTIČARKE\n\n"
            "TEJA BELAK\n\n"
            "Teja Belak je ena najpomembnejših slovenskih gimnastičark. Nastopila je na "
            "Olimpijskih igrah v Riu de Janeiru leta 2016 in v Tokiu leta 2020, s čimer je "
            "postala ena redkih slovenskih gimnastičark z dvema olimpijskima nastopoma. Je "
            "večkratna državna prvakinja in je Slovenijo zastopala na Evropskih in Svetovnih "
            "prvenstvih.\n\n"
            "TJAŠA KYSSELEF\n\n"
            "Tjaša Kysselef je ena najdlje delujočih slovenskih tekmovalk na mednarodni ravni. "
            "Nastopala je na številnih Evropskih prvenstvih, Svetovnih prvenstvih in tekmovanjih "
            "serije World Cup.\n\n"
            "LUCIJA HRIBAR\n\n"
            "Lucija Hribar je trenutno najbolj uspešna gimnastičarka. Leta 2024 se je na "
            "olimpijskih igrah v Parizu uvrstila v finale mnogoboja, kar je za Slovenijo "
            "najboljši dosežek doslej."
        ),
    },
    {
        "id": 19,
        "title": "Tekmovalni sistem GZS",
        "language": "sl",
        "category": "Slovenija",
        "content": (
            "TEKMOVALNI SISTEM GZS\n\n"
            "NIVOJI TEKMOVANJ V SLOVENIJI\n\n"
            "V Sloveniji so tekmovanja v ženski športni gimnastiki organizirana v dveh osnovnih "
            "programih: stopenjskem programu in mednarodnem programu.\n\n"
            "STOPENJSKI PROGRAM\n\n"
            "Stopenjski program je namenjen predvsem dekletom, ki se z gimnastiko ukvarjajo "
            "rekreativno ali manj intenzivno. Vanj so vključene tudi mlajše začetnice, ki šele "
            "začenjajo svojo gimnastično pot. Tekmovanja potekajo po prilagojenem nacionalnem "
            "pravilniku, ki ga pripravlja Gimnastična zveza Slovenije. Ta pravilnik poenostavlja "
            "zahteve glede sestav, težavnosti in izvedbe ter omogoča postopno in varno usvajanje "
            "osnovnih gimnastičnih prvin. Stopenjski program vključuje več stopenj (1 do 5 ter "
            "nadalje tudi višje stopnje), pri čemer se zahtevnost postopoma povečuje. Sistem je "
            "zasnovan tako, da tekmovalkam omogoča razvoj osnov, občutek za tekmovanje ter "
            "postopno napredovanje.\n\n"
            "MEDNARODNI PROGRAM\n\n"
            "Mednarodni program je namenjen tekmovalkam, ki trenirajo gimnastiko bolj resno "
            "(pogosto že od mlajših kategorij dalje). V tem programu tekmovalke nastopajo po "
            "pravilih Mednarodne gimnastične zveze (FIG), ki določa sestavo vaj, vrednotenje "
            "težavnosti (D-ocena) in izvedbe (E-ocena). Ta pravilnik se uporablja na vseh "
            "najvišjih tekmovanjih, kot so svetovna prvenstva in olimpijske igre, ter zagotavlja "
            "enotne standarde ocenjevanja na mednarodni ravni.\n\n"
            "PREHAJANJE MED PROGRAMOMA\n\n"
            "V praksi tekmovalke pogosto začnejo v stopenjskem programu, kjer pridobijo osnovno "
            "znanje. Tiste, ki nadaljujejo z bolj intenzivnim treningom, kasneje preidejo v "
            "mednarodni program. Tekmovalke, ki že od začetka trenirajo na višji ravni, pa lahko "
            "v mednarodnem programu nastopajo že v mlajših kategorijah."
        ),
    },
]

# ============================================================
# RAG SETUP — runs once and is cached
# ============================================================
@st.cache_resource
def setup_rag():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embeddings = SentenceTransformerEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L6-v2"
    )
    langchain_docs = []
    for doc in DOCUMENTS:
        for chunk in splitter.split_text(doc["content"]):
            langchain_docs.append(Document(
                page_content=chunk,
                metadata={
                    "title": doc["title"],
                    "language": doc["language"],
                    "category": doc["category"],
                }
            ))
    return Chroma.from_documents(langchain_docs, embeddings)


# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 22px 0 10px 0;">
        <div style="color:white; font-size:1.15em; font-weight:700; letter-spacing:0.5px;">Gymnastics Portal</div>
        <div style="color:#8B6FB5; font-size:0.78em; margin-top:3px; letter-spacing:1.2px; text-transform:uppercase;">Gimnastika</div>
    </div>
    <hr style="border-color:#ffffff18; margin:6px 0 12px 0;">
    """, unsafe_allow_html=True)

    for label, key in [("Home", "home"), ("About", "about_sport")]:
        if st.button(label, use_container_width=True, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

    with st.expander("Disciplines"):
        if st.button("Overview", use_container_width=True, key="nav_disciplines"):
            st.session_state.page = "disciplines"
            st.rerun()
        for label, key in [
            ("Artistic",    "disc_artistic"),
            ("Rhythmic",    "disc_rhythmic"),
            ("Aerobic",     "disc_aerobic"),
            ("Trampoline",  "disc_trampoline"),
            ("Tumbling",    "disc_tumbling"),
            ("Acrobatic",   "disc_acrobatic"),
            ("Parkour",     "disc_parkour"),
        ]:
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()

    for label, key in [("History", "history"), ("Judging System", "judging"), ("Competitions", "competitions")]:
        if st.button(label, use_container_width=True, key=f"nav_{key}"):
            st.session_state.page = key
            st.rerun()

    st.markdown("<hr style='border-color:#ffffff22; margin:10px 0;'>", unsafe_allow_html=True)
    if st.button("Search / Iskanje", use_container_width=True, key="nav_search"):
        st.session_state.page = "search"
        st.rerun()

    st.markdown("<hr style='border-color:#ffffff22; margin:10px 0;'>", unsafe_allow_html=True)
    if st.button("About this Portal", use_container_width=True, key="nav_about_portal"):
        st.session_state.page = "about_portal"
        st.rerun()

    st.markdown(f"""
    <div class="sidebar-logo-bottom">
        <div style="color:#5B4080; font-size:11px; letter-spacing:0.6px; text-transform:uppercase; margin-bottom:16px; line-height:1.6;">
            Based on FIG Code of Points<br>2025–2028
        </div>
        <img src="data:image/png;base64,{LOGO_B64}" width="130">
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# LOAD VECTOR STORE (cached after first load)
# ============================================================
with st.spinner("Loading knowledge base..."):
    vectorstore = setup_rag()


# ============================================================
# HOME PAGE
# ============================================================
# HOME
# ============================================================
def show_home():
    _hero = GRED_IMG or BACKGROUND_IMG
    _tile = BACKGROUND_IMG or GRED_IMG
    if _hero:
        st.markdown(f"""
        <div style="position:relative;overflow:hidden;border-radius:6px;min-height:220px;margin-bottom:32px;text-align:center;">
            <div style="position:absolute;inset:0;background:url({_hero});background-size:cover;background-position:center;filter:blur(4px);transform:scale(1.08);"></div>
            <div style="position:absolute;inset:0;background:rgba(10,2,30,0.62);"></div>
            <div style="position:relative;z-index:1;padding:56px 40px 44px 40px;">
                <h1 style="color:white;font-size:2.8em;font-weight:800;text-transform:uppercase;letter-spacing:1px;margin:0 0 14px 0;">Gymnastics Portal</h1>
                <p style="color:rgba(255,255,255,0.82);font-size:1.1em;margin:0;">A beginner-friendly guide to gymnastics — from the foundations to the judging system</p>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hero-banner">
            <h1>Gymnastics Portal</h1>
            <p>A beginner-friendly guide to gymnastics — from the foundations to the judging system</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Explore the Portal")
    st.markdown("Select a section below or use the sidebar to navigate.")

    sections = [
        ("About",          "about_sport",   "What gymnastics is and what it takes to be a gymnast."),
        ("Disciplines",    "disciplines",   "All seven gymnastics disciplines and their events."),
        ("History",        "history",       "Origins of the sport and notable athletes."),
        ("Judging System", "judging",       "How WAG scoring works based on the FIG Code of Points."),
        ("Competitions",   "competitions",  "Competition formats from club level to the Olympics."),
        ("Search",         "search",        "Ask any question in English or Slovenian."),
    ]

    FALLBACK_COLORS = ["#5B21B6", "#0F766E", "#1E0A45", "#7C3AED", "#065F46", "#312E81"]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (label, page_key, desc) in enumerate(sections):
        with cols[i % 3]:
            if _tile:
                st.markdown(f"""
                <div style="position:relative;overflow:hidden;border-radius:4px;height:200px;margin-bottom:4px;">
                    <div style="position:absolute;inset:0;background:url({_tile});background-size:cover;background-position:center;filter:blur(3px);transform:scale(1.08);"></div>
                    <div style="position:absolute;inset:0;background:rgba(10,2,30,0.55);"></div>
                    <div style="position:relative;z-index:1;padding:24px 24px 20px 24px;">
                        <div style="color:rgba(255,255,255,0.65);font-size:21px;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">{label.upper()}</div>
                        <div style="color:rgba(255,255,255,0.88);font-size:30px;line-height:1.4;">{desc}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                color = FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
                st.markdown(f"""
                <div class="metro-tile" style="background:{color};">
                    <h4>{label}</h4>
                    <p>{desc}</p>
                </div>""", unsafe_allow_html=True)
            if st.button("Open", use_container_width=True, key=f"home_{page_key}"):
                st.session_state.page = page_key
                st.rerun()
            st.markdown(" ")


# ============================================================
# ABOUT (THE SPORT)
# ============================================================
def show_about_sport():
    st.markdown('<h2 class="section-header">About Gymnastics</h2>', unsafe_allow_html=True)

    # Section 1: text left, minikeli na gredi right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div style="color:#7C3AED;font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">The Sport</div>', unsafe_allow_html=True)
        st.markdown("### What is Gymnastics?")
        st.markdown("""
Gymnastics is a sport that combines strength, flexibility, balance, coordination, and movement technique across a range of disciplines and apparatus.

Its origins trace back to ancient Greece, where gymnastic exercises were part of physical and military education. The modern sport took shape in 19th-century Europe, largely through the work of Friedrich Ludwig Jahn in Germany, widely regarded as the "father of gymnastics." Gymnastics was included in the first modern Olympic Games in Athens in 1896 and has been part of every Summer Olympics since.

The sport is governed internationally by the **Fédération Internationale de Gymnastique (FIG)**, founded in 1881 and headquartered in Lausanne, Switzerland. FIG establishes the competition rules through the Code of Points, organizes the World Championships, and oversees the Olympic gymnastics program. The organization has member federations in over 140 countries.

Athletes typically begin training in early childhood and progress through national competition systems toward international competition. The highest level of competition is the Olympic Games.
""")
    with col2:
        st.image("minikeli na gredi.jpg", use_container_width=True)

    st.markdown("---")

    # Section 2: children left, text right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image("children.jpg", use_container_width=True)
    with col2:
        st.markdown('<div style="color:#7C3AED;font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">The Athlete</div>', unsafe_allow_html=True)
        st.markdown("### What Does It Take to Be a Gymnast?")
        st.markdown("""
Gymnastics is a sport that requires a combination of physical and mental qualities developed over many years of training.

**Training volume:** Elite gymnasts typically train 7 to 8 hours per day, which amounts to roughly 40 to 50 hours per week. At younger and lower levels, training is less intensive, but consistent practice over years is necessary to develop the required skills.

**Starting age:** Most competitive gymnasts begin training between the ages of 3 and 6. At the elite international level, gymnasts typically reach their peak competitive years between the ages of 15 and 24 in the women's artistic discipline.

**Physical qualities:** The sport requires flexibility, strength relative to body weight, coordination, spatial awareness, and the ability to learn and repeat complex movement patterns precisely.

**Mental demands:** Gymnasts must manage performance under pressure, overcome fear when learning new skills, and maintain consistency across multiple events in a single competition. Mental preparation is considered as important as physical training at the elite level.

**Education:** Due to the high number of training hours, elite gymnasts are frequently homeschooled or follow adapted educational programs. This allows them to maintain their training schedule while fulfilling their academic requirements.
""")


# ============================================================
# DISCIPLINES
# ============================================================
def _disc_tile(label, subtitle, img_data, fallback_color, height="160px", bg_position="center"):
    if img_data:
        return f"""
        <div style="position:relative;overflow:hidden;border-radius:4px;min-height:{height};margin-bottom:4px;">
            <div style="position:absolute;inset:0;background:url({img_data});background-size:cover;background-position:{bg_position};filter:blur(3px);transform:scale(1.08);"></div>
            <div style="position:absolute;inset:0;background:rgba(10,2,30,0.52);"></div>
            <div style="position:relative;z-index:1;padding:28px 24px 22px 24px;">
                <div style="color:white;font-size:34px;font-weight:800;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
                <div style="color:rgba(255,255,255,0.82);font-size:30px;margin-top:8px;">{subtitle}</div>
            </div>
        </div>"""
    return f"""
        <div style="background:{fallback_color};padding:28px 24px 22px 24px;border-radius:4px;min-height:{height};margin-bottom:4px;">
            <div style="color:white;font-size:34px;font-weight:800;text-transform:uppercase;letter-spacing:0.5px;">{label}</div>
            <div style="color:rgba(255,255,255,0.68);font-size:30px;margin-top:8px;">{subtitle}</div>
        </div>"""


def show_disciplines():
    st.markdown('<h2 class="section-header">Disciplines</h2>', unsafe_allow_html=True)
    st.markdown("Gymnastics comprises **seven official disciplines** governed by FIG. Select a discipline below to explore its events and format.")

    # Featured: ARTISTIC
    if DISC_IMGS["artistic"]:
        st.markdown(f"""
        <div style="position:relative;overflow:hidden;border-radius:4px;min-height:200px;margin:24px 0 8px 0;">
            <div style="position:absolute;inset:0;background:url({DISC_IMGS['artistic']});background-size:cover;background-position:center 20%;filter:blur(3px);transform:scale(1.08);"></div>
            <div style="position:absolute;inset:0;background:rgba(10,2,30,0.55);"></div>
            <div style="position:relative;z-index:1;padding:44px 40px 36px 40px;">
                <div style="color:rgba(255,255,255,0.65);font-size:21px;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Most Widely Known</div>
                <div style="color:white;font-size:2.4em;font-weight:800;text-transform:uppercase;letter-spacing:0.3px;margin-bottom:12px;">Artistic</div>
                <div style="color:rgba(255,255,255,0.82);font-size:30px;">Women's Artistic (WAG) &amp; Men's Artistic (MAG) — 4 and 6 apparatus respectively</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background:linear-gradient(135deg,#3B0764,#7C3AED);padding:44px 40px 36px 40px;border-radius:4px;margin:24px 0 8px 0;">
            <div style="color:rgba(255,255,255,0.6);font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">Most Widely Known</div>
            <div style="color:white;font-size:2.4em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Artistic</div>
            <div style="color:rgba(255,255,255,0.78);font-size:30px;">Women's Artistic (WAG) &amp; Men's Artistic (MAG) — 4 and 6 apparatus respectively</div>
        </div>""", unsafe_allow_html=True)

    if st.button("Explore Artistic Gymnastics →", key="hub_disc_artistic"):
        st.session_state.page = "disc_artistic"
        st.rerun()

    st.markdown(" ")
    remaining = [
        ("RHYTHMIC",   "disc_rhythmic",   "rhythmic",   "#065F46", "Individual &amp; Group Events",   "center"),
        ("AEROBIC",    "disc_aerobic",    "aerobic",    "#1E0A45", "5 Competition Formats",            "center"),
        ("TRAMPOLINE", "disc_trampoline", "trampoline", "#1E3A5F", "Trampoline &amp; Double Mini",     "center"),
        ("TUMBLING",   "disc_tumbling",   "tumbling",   "#3B1F6B", "Power Tumbling",                   "center 70%"),
        ("ACROBATIC",  "disc_acrobatic",  "acrobatic",  "#0C3547", "Pairs &amp; Groups",               "center"),
        ("PARKOUR",    "disc_parkour",    "parkour",    "#312E81", "Speed, Freestyle &amp; Team",       "center"),
    ]
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (label, key, img_key, color, subtitle, bg_pos) in enumerate(remaining):
        with cols[i % 3]:
            st.markdown(_disc_tile(label, subtitle, DISC_IMGS.get(img_key, ""), color, bg_position=bg_pos), unsafe_allow_html=True)
            if st.button("Explore →", key=f"hub_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()
            st.markdown(" ")


def _disc_back():
    if st.button("← Back to Disciplines", key="back_disc_btn"):
        st.session_state.page = "disciplines"
        st.rerun()
    st.markdown(" ")


def show_disc_artistic():
    st.markdown("""<div style="background:linear-gradient(135deg,#3B0764,#7C3AED);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Artistic Gymnastics</div>
    <div style="font-size:30px;opacity:0.82;">Women's Artistic (WAG) &amp; Men's Artistic (MAG)</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Artistic Gymnastics is the most widely recognized FIG discipline. Women compete on four apparatus; men on six.")
    st.markdown("### Women's Artistic Gymnastics (WAG)")
    with st.expander("Vault"):
        st.markdown("The gymnast sprints approximately 25 meters down the runway, takes off from a springboard, pushes off the vaulting table, and executes a skill in the air before landing. The flight phase lasts approximately 1–2 seconds. Scores are based on the difficulty of the vault and the quality of execution and landing.")
    with st.expander("Uneven Bars"):
        st.markdown("Two horizontal bars set at different heights — high bar at approximately 2.50 m, low bar at 1.70 m. The gymnast performs a continuous routine of swings, releases, and catches. Evaluation focuses on swing technique, release skills, and body position.")
    with st.expander("Balance Beam"):
        st.markdown("The beam is 5 meters long and 10 centimeters wide, raised 125 centimeters above the floor. Gymnasts perform a routine of 70–90 seconds combining jumps, turns, acrobatic elements, and choreography. Any step, fall, or loss of balance results in a deduction.")
    with st.expander("Floor Exercise"):
        st.markdown("The routine is performed on a 12×12 meter sprung floor to music, lasting 70–90 seconds. It includes tumbling passes, jumps, turns, and choreographic sequences. Going outside the boundary lines results in a deduction.")
    st.markdown("### Men's Artistic Gymnastics (MAG)")
    with st.expander("Floor"):
        st.markdown("Tumbling passes and strength elements are performed on a 12×12 meter sprung floor, without musical accompaniment at the international level.")
    with st.expander("Pommel Horse"):
        st.markdown("The gymnast performs continuous circular and scissor movements on a leather horse with two handles, supporting the body on the arms throughout. Any stop in movement or contact with the leg results in a deduction.")
    with st.expander("Still Rings"):
        st.markdown("Two rings suspended from cables at 2.75 meters. The gymnast performs swing elements and static strength holds — such as the iron cross or planche — while minimizing movement in the rings themselves.")
    with st.expander("Vault — MAG"):
        st.markdown("Performed on the same apparatus as in the women's discipline, at higher difficulty levels.")
    with st.expander("Parallel Bars"):
        st.markdown("Two bars set at equal height (195 cm), approximately 42 cm apart. Gymnasts perform swinging, release, and strength elements both between and above the bars.")
    with st.expander("Horizontal Bar"):
        st.markdown("A single bar set at 280 cm. Gymnasts perform giant swings and release skills, briefly releasing the bar entirely before catching it again and finishing with a dismount.")


def show_disc_rhythmic():
    st.markdown("""<div style="background:linear-gradient(135deg,#064E3B,#0D9488);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Rhythmic Gymnastics</div>
    <div style="font-size:30px;opacity:0.82;">Individual &amp; Group Events</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Rhythmic Gymnastics combines movement drawn from dance and ballet with the technical manipulation of hand apparatus. Competitions are held in individual and group formats.")
    with st.expander("Individual"):
        st.markdown("""Gymnasts compete with five types of hand apparatus:
- **Rope:** Jumping, throwing, and rotating the rope.
- **Hoop:** Rolling along the body, throwing, rotating, and passing through.
- **Ball:** Rolling on the body, bouncing, and throwing.
- **Clubs:** Throwing and catching skills performed with two clubs simultaneously.
- **Ribbon:** Continuous movement that draws shapes in the air; contact with the floor results in a deduction.""")
    with st.expander("Group"):
        st.markdown("Five gymnasts perform together, either with identical apparatus or a combination of two different types. Russia, Bulgaria, Israel, and Spain have historically achieved the strongest international results.")


def show_disc_aerobic():
    st.markdown("""<div style="background:linear-gradient(135deg,#1E0A45,#4C1D95);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Aerobic Gymnastics</div>
    <div style="font-size:30px;opacity:0.82;">5 Competition Formats</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Aerobic Gymnastics routines last **60–90 seconds** and consist of continuous, high-intensity movement performed to music. Athletes demonstrate strength, flexibility, coordination, and cardiorespiratory endurance.")
    with st.expander("Individual Men"):
        st.markdown("A solo male athlete performs a high-intensity aerobic routine to music.")
    with st.expander("Individual Women"):
        st.markdown("A solo female athlete performs a high-intensity aerobic routine to music.")
    with st.expander("Mixed Pairs"):
        st.markdown("One male and one female athlete perform a synchronized aerobic routine together.")
    with st.expander("Trios"):
        st.markdown("Three athletes perform a synchronized aerobic routine.")
    with st.expander("Groups"):
        st.markdown("Six athletes perform a synchronized aerobic routine together.")


def show_disc_trampoline():
    st.markdown("""<div style="background:linear-gradient(135deg,#1E3A5F,#2563EB);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Trampoline</div>
    <div style="font-size:30px;opacity:0.82;">Trampoline &amp; Double Mini Trampoline</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Trampoline Gymnastics encompasses two related disciplines performed on different apparatus.")
    with st.expander("Trampoline"):
        st.markdown("Athletes perform a routine of **10 consecutive acrobatic skills** on a trampoline, reaching heights of up to 8–10 meters. Each skill must be different. Routines are evaluated on difficulty, execution quality, and total **time of flight** — how long the athlete stays in the air across the entire routine.")
    with st.expander("Double Mini Trampoline"):
        st.markdown("Athletes use a smaller trampoline with a run-up. The routine consists of a mount and a dismount skill, combining the run-up speed with aerial acrobatics.")


def show_disc_tumbling():
    st.markdown("""<div style="background:linear-gradient(135deg,#3B1F6B,#6D28D9);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Tumbling</div>
    <div style="font-size:30px;opacity:0.82;">Power Tumbling on a Sprung Runway</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Tumbling (also called Power Tumbling) is performed on a 25-meter sprung runway. There is no trampoline — the energy comes entirely from the athlete and the runway itself.")
    with st.expander("Tumbling"):
        st.markdown("Athletes perform a pass of **8 consecutive acrobatic skills** down a sprung runway, combining power, speed, and aerial technique. Routines are judged on difficulty and execution.")


def show_disc_acrobatic():
    st.markdown("""<div style="background:linear-gradient(135deg,#0C3547,#0369A1);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Acrobatic Gymnastics</div>
    <div style="font-size:30px;opacity:0.82;">Pairs &amp; Groups — Balance &amp; Dynamic Elements</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Acrobatic Gymnastics features athletes competing in partnerships, combining balance and dynamic elements performed to music.")
    with st.expander("Women's Pairs"):
        st.markdown("Two female athletes perform balance and dynamic elements together, evaluated on difficulty, execution, and artistry.")
    with st.expander("Men's Pairs"):
        st.markdown("Two male athletes perform balance and dynamic elements together.")
    with st.expander("Mixed Pairs"):
        st.markdown("One male and one female athlete perform together. The male typically acts as the base while the female performs aerial and balance skills.")
    with st.expander("Women's Groups"):
        st.markdown("Three female athletes perform together, combining balance pyramids and dynamic throws.")
    with st.expander("Men's Groups"):
        st.markdown("Four male athletes perform together, building the largest and most complex formations in the discipline.")


def show_disc_parkour():
    st.markdown("""<div style="background:linear-gradient(135deg,#1E1B4B,#4338CA);color:white;padding:48px 42px 38px 42px;border-radius:4px;margin-bottom:32px;">
    <div style="font-size:14px;text-transform:uppercase;letter-spacing:2px;opacity:0.6;margin-bottom:10px;">Discipline</div>
    <div style="font-size:2.6em;font-weight:800;text-transform:uppercase;margin-bottom:12px;">Parkour</div>
    <div style="font-size:30px;opacity:0.82;">Speed, Freestyle &amp; Freestyle Team</div>
    </div>""", unsafe_allow_html=True)
    _disc_back()
    st.markdown("Parkour was officially added to FIG in 2018. Athletes navigate obstacle courses with speed, creativity, and control.")
    with st.expander("Speed Run"):
        st.markdown("Athletes race through a standardized obstacle course as fast as possible. The winner is determined purely by time.")
    with st.expander("Freestyle"):
        st.markdown("Athletes perform creative lines through an obstacle course, judged on difficulty, execution, and artistic expression.")
    with st.expander("Freestyle Team"):
        st.markdown("Teams of athletes perform a synchronized freestyle routine together, judged on coordination, difficulty, and creativity.")


# ============================================================
# HISTORY
# ============================================================
def show_history():
    st.markdown('<h2 class="section-header">History</h2>', unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    # ── Section 1: Origins — text left, image right ──
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""<div style="padding: 20px 0;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:2px; color:#7C3AED; font-weight:600; margin-bottom:14px;">The Beginning</div>
            <div style="font-size:28px; font-weight:700; color:#0C0219; line-height:1.25; margin-bottom:20px;">Origins of the Sport</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
The origins of gymnastics trace back to ancient Greece, where gymnastic exercises were part of physical and military education. The modern sport took shape in 19th-century Europe, largely through the work of **Friedrich Ludwig Jahn** in Germany, widely regarded as the "father of gymnastics."

Gymnastics was included in the first modern **Olympic Games in Athens in 1896** and has been part of every Summer Olympics since. The sport is governed internationally by the **Fédération Internationale de Gymnastique (FIG)**, founded in 1881 and headquartered in Lausanne, Switzerland.
""")
    with col2:
        st.image("history.jpg", use_container_width=True)

    st.markdown("<div style='margin: 64px 0; border-top: 1px solid #E5E7EB;'></div>", unsafe_allow_html=True)

    # ── Section 2: Notable Gymnasts — image left, text right ──
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image("simone biles.jpg", use_container_width=True)
    with col2:
        st.markdown("""<div style="padding: 20px 0;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:2px; color:#7C3AED; font-weight:600; margin-bottom:14px;">Legends of the Sport</div>
            <div style="font-size:28px; font-weight:700; color:#0C0219; line-height:1.25; margin-bottom:20px;">Notable Gymnasts in History</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
**Larisa Latynina (USSR)** competed between 1956 and 1964, winning 18 Olympic medals — 9 gold, 5 silver, and 4 bronze. This remained the record for most Olympic medals by any athlete for nearly 50 years.

**Věra Čáslavská (Czechoslovakia)** won 7 Olympic gold medals across the 1964 and 1968 Games.

**Nadia Comaneci (Romania)** became the first gymnast in Olympic history to receive a perfect score of 10.0, at the 1976 Montreal Games at age 14.

**Simone Biles (USA)** has accumulated more World Championship medals than any other gymnast in history. She has introduced multiple skills of unprecedented difficulty, several of which now carry her name in the Code of Points.
""")

    st.markdown("<div style='margin: 64px 0; border-top: 1px solid #E5E7EB;'></div>", unsafe_allow_html=True)

    # ── Section 3: Slovenia — text left, image right ──
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""<div style="padding: 20px 0;">
            <div style="font-size:12px; text-transform:uppercase; letter-spacing:2px; color:#7C3AED; font-weight:600; margin-bottom:14px;">Domača scena</div>
            <div style="font-size:28px; font-weight:700; color:#0C0219; line-height:1.25; margin-bottom:20px;">Gimnastika v Sloveniji</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
Gimnastična zveza Slovenije (GZS) je nacionalna krovska organizacija za gimnastiko v Sloveniji. Je članica Mednarodne gimnastične zveze (FIG) in Evropske gimnastične zveze (UEG).

Po osamosvojitvi leta 1991 je Slovenija začela nastopati na mednarodnih tekmovanjih pod lastno zastavo. **Teja Belak** je nastopila na Olimpijskih igrah v Riu 2016 in Tokiu 2020. **Lucija Hribar** se je leta 2024 v Parizu uvrstila v finale mnogoboja — najboljši slovenski dosežek doslej.
""")
    with col2:
        st.image("teja belak.jpg", use_container_width=True)


# ============================================================
# JUDGING SYSTEM
# ============================================================
def show_judging():
    def _card(img, content, label=None):
        lbl = f'<div style="color:rgba(255,255,255,0.65);font-size:21px;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">{label}</div>' if label else ""
        if img:
            return (f'<div style="position:relative;overflow:hidden;border-radius:6px;margin-bottom:14px;">'
                    f'<div style="position:absolute;inset:0;background:url({img});background-size:cover;background-position:center;filter:blur(4px);transform:scale(1.08);"></div>'
                    f'<div style="position:absolute;inset:0;background:rgba(8,2,22,0.20);"></div>'
                    f'<div style="position:relative;z-index:1;padding:24px 28px;">{lbl}{content}</div>'
                    f'</div>')
        return f'<div style="background:#1E0A45;border-radius:6px;padding:24px 28px;margin-bottom:14px;">{lbl}{content}</div>'

    def _half(img, content, label=None):
        lbl = f'<div style="color:rgba(255,255,255,0.65);font-size:21px;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;">{label}</div>' if label else ""
        if img:
            return (f'<div style="position:relative;overflow:hidden;border-radius:6px;flex:1;">'
                    f'<div style="position:absolute;inset:0;background:url({img});background-size:cover;background-position:center;filter:blur(4px);transform:scale(1.08);"></div>'
                    f'<div style="position:absolute;inset:0;background:rgba(8,2,22,0.20);"></div>'
                    f'<div style="position:relative;z-index:1;padding:24px 28px;">{lbl}{content}</div>'
                    f'</div>')
        return f'<div style="flex:1;background:#1E0A45;border-radius:6px;padding:24px 28px;">{lbl}{content}</div>'

    def _row(left_html, right_html):
        return f'<div style="display:flex;gap:14px;margin-bottom:14px;">{left_html}{right_html}</div>'

    B = BG_IMGS  # shorthand; cycles: B[0..3]
    TXT = "color:rgba(255,255,255,0.92);font-size:27px;line-height:1.8;"
    TBL = "width:100%;border-collapse:collapse;margin-top:8px;"
    TH  = "background:rgba(0,0,0,0.30);color:white;padding:8px 12px;text-align:left;font-size:25px;"
    TD  = "padding:8px 12px;font-size:25px;border-top:1px solid rgba(255,255,255,0.10);color:rgba(255,255,255,0.88);"

    st.markdown('<h2 class="section-header">Judging System</h2>', unsafe_allow_html=True)
    st.markdown("Based on the **FIG Code of Points 2025–2028** for Women's Artistic Gymnastics.")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Scoring Overview", "D Score", "E Score", "Judging Panel", "Vault & Bars", "Beam & Floor"
    ])

    with tab1:
        st.markdown(
            _card(B[0],
                f'<div style="color:white;font-size:1.2em;font-weight:700;margin-bottom:12px;">Final Score = D Score + E Score − Neutral Deductions</div>'
                f'<div style="{TXT}">Women\'s Artistic Gymnastics uses an <strong>open-ended scoring system</strong> introduced after the 2004 Olympic Games. Scores are not capped at 10.0.</div>',
                "The Formula") +
            _row(
                _half(B[1],
                    f'<div style="{TXT}">Has <strong>no maximum limit.</strong> A gymnast who includes more difficult skills and fulfills more composition requirements earns a higher D score.</div>',
                    "Difficulty Score (D)"),
                _half(B[2],
                    f'<div style="{TXT}">Starts at a maximum of <strong>10.0.</strong> Judges deduct for technical and artistic errors throughout the routine.</div>',
                    "Execution Score (E)")
            ) +
            _card(B[3],
                f'<div style="{TXT}">If two gymnasts finish with the same final score, the gymnast with the <strong>higher D Score</strong> is ranked higher.</div>',
                "Tiebreaker Rule"),
            unsafe_allow_html=True)

    with tab2:
        st.markdown(
            _card(B[1],
                f'<div style="{TXT};margin-bottom:12px;">Every skill is assigned a letter A to I. On floor, beam, and bars only the <strong>8 highest-valued elements</strong> count. On vault, the score is the value of the single skill performed.</div>'
                f'<table style="{TBL}"><tr>'
                + "".join(f'<th style="{TH}">{l}</th>' for l in ["Letter","A","B","C","D","E","F","G","H","I"])
                + f'</tr><tr><td style="{TD}font-weight:600;">Value</td>'
                + "".join(f'<td style="{TD}">0.{v}</td>' for v in range(1,10))
                + '</tr></table>',
                "Element Values — A to I") +
            _row(
                _half(B[2],
                    f'<div style="{TXT}">Each fulfilled requirement adds <strong>0.5 points</strong> to the D score. Maximum of <strong>2.0 points</strong> available. Missing a requirement simply means those points are not earned.</div>',
                    "Composition Requirements (CR)"),
                _half(B[3],
                    f'<div style="{TXT}">Gymnasts can earn bonus tenths (<strong>0.10 or 0.20</strong>) by directly connecting certain elements in sequence.</div>',
                    "Connection Value (CV)")
            ),
            unsafe_allow_html=True)

    with tab3:
        st.markdown(
            _card(B[2],
                f'<div style="{TXT}">The E Score begins at <strong>10.0</strong> for every routine. Five judges evaluate execution independently. The highest and lowest scores are removed; the final E score is the <strong>average of the remaining three.</strong></div>',
                "How E Score Works") +
            _row(
                _half(B[3],
                    f'<table style="{TBL}"><tr><th style="{TH}">Severity</th><th style="{TH}">Deduction</th></tr>'
                    f'<tr><td style="{TD}">Small error</td><td style="{TD}font-weight:700;">0.10</td></tr>'
                    f'<tr><td style="{TD}">Medium error</td><td style="{TD}font-weight:700;">0.30</td></tr>'
                    f'<tr><td style="{TD}">Large error</td><td style="{TD}font-weight:700;">0.50</td></tr>'
                    f'<tr><td style="{TD}">Fall</td><td style="{TD}font-weight:700;color:#fca5a5;">1.00</td></tr></table>'
                    f'<div style="color:rgba(255,255,255,0.65);font-size:24px;margin-top:10px;">Max per element: 0.80 · Falls always 1.00</div>',
                    "Deduction Scale"),
                _half(B[0],
                    f'<div style="{TXT};line-height:2;">Bent knees — 0.10–0.30<br>Separated legs — 0.10–0.30<br>Flexed feet — 0.10–0.30<br>Insufficient height — 0.10–0.30<br>Landing steps — 0.10–0.30<br>Loss of balance on beam — 0.10–0.50</div>',
                    "Common Errors")
            ),
            unsafe_allow_html=True)

    with tab4:
        st.markdown(
            _row(
                _half(B[3],
                    f'<div style="{TXT}">Identify every skill, verify Composition Requirements, and calculate the Difficulty Score. They work independently and compare results.</div>',
                    "D Panel — 2 Judges"),
                _half(B[0],
                    f'<div style="{TXT}">Each assesses execution independently. The highest and lowest scores are dropped; the average of the remaining three is the final E score.</div>',
                    "E Panel — 5 Judges")
            ) +
            _row(
                _half(B[1],
                    f'<div style="{TXT}">Gives the starting signal, monitors neutral deductions (e.g. time violations), and ensures FIG regulations are followed.</div>',
                    "Referee"),
                _half(B[2],
                    f'<div style="{TXT}">A coach may challenge the <strong>D score only.</strong> E score decisions are final. The inquiry fee is refunded if the challenge is successful.</div>',
                    "Inquiries")
            ),
            unsafe_allow_html=True)

    with tab5:
        st.markdown(
            _row(
                _half(B[0],
                    f'<div style="{TXT};line-height:2;">The D score is the predetermined value of the vault.<br><br>'
                    f'Landing steps — 0.10 / 0.30<br>Fall on landing — <strong style="color:#fca5a5;">1.00</strong><br>'
                    f'Flexed feet during flight — 0.10 / 0.30<br>Bent knees — 0.10 / 0.30<br>'
                    f'Low or under-rotated vault — 0.30 / 0.50<br>Hands touching mat — 0.50</div>'
                    f'<div style="color:rgba(255,255,255,0.65);font-size:24px;margin-top:12px;">In vault finals, two vaults are performed and the scores averaged.</div>',
                    "Vault"),
                _half(B[3],
                    f'<div style="{TXT};line-height:2;">Up to 8 counted elements.<br><br>'
                    f'Bent knees / piked body — 0.10 / 0.30<br>Flexed feet — 0.10 / 0.30<br>'
                    f'Insufficient height on release — 0.10 / 0.30<br>Landing steps — 0.10 / 0.30<br>'
                    f'Fall — <strong style="color:#fca5a5;">1.00</strong></div>'
                    f'<div style="color:rgba(255,255,255,0.65);font-size:24px;margin-top:12px;">After a fall, gymnast has 30 seconds to remount or the routine is terminated.</div>',
                    "Uneven Bars")
            ),
            unsafe_allow_html=True)

    with tab6:
        st.markdown(
            _row(
                _half(B[1],
                    f'<div style="{TXT};line-height:2;">Routine must last <strong>70–90 seconds.</strong> Each full second over: 0.10 neutral deduction.<br><br>'
                    f'Small loss of balance — 0.10<br>Significant loss of balance — 0.30<br>'
                    f'Hands touch beam — 0.50<br>Fall — <strong style="color:#fca5a5;">1.00</strong><br>'
                    f'Bent knees — 0.10 / 0.30<br>Landing step — 0.10 / 0.30</div>'
                    f'<div style="color:rgba(255,255,255,0.65);font-size:24px;margin-top:12px;">After a fall, gymnast has 10 seconds to remount or the routine is terminated.</div>',
                    "Balance Beam"),
                _half(B[2],
                    f'<div style="{TXT};line-height:2;">Routine must last <strong>70–90 seconds,</strong> performed to music. Each full second over: 0.10 neutral deduction.<br><br>'
                    f'Landing steps — 0.10 / 0.30<br>Bent knees or legs apart on salto — 0.10 / 0.30<br>'
                    f'Flexed feet — 0.10 / 0.30<br>Out of bounds — <strong>0.30 per foot</strong></div>'
                    f'<div style="color:rgba(255,255,255,0.65);font-size:24px;margin-top:12px;">Floor is 12×12 m. A foot on the line is fine; fully outside is penalized.</div>',
                    "Floor Exercise")
            ),
            unsafe_allow_html=True)



# ============================================================
# COMPETITIONS
# ============================================================
def show_competitions():
    st.markdown('<h2 class="section-header">Competitions</h2>', unsafe_allow_html=True)

    # Section 1: text left, competition.jpg right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div style="color:#7C3AED;font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Levels of Competition</div>', unsafe_allow_html=True)
        st.markdown("### Types of Competitions")
        st.markdown("""
Gymnastics competitions are organized at several levels, from local club events to the Olympic Games.

**Olympic Games:** Held every four years — the highest level of gymnastics competition. Countries qualify based on results at the preceding World Championships and qualification events.

**World Championships:** Organized annually by FIG — the most important competition outside the Olympic cycle. Athletes compete in team, individual all-around, and individual apparatus events.

**World Cup Series:** A series of individual apparatus competitions held in different countries throughout the year. Athletes accumulate points toward an overall World Cup ranking.

**Continental Championships:** In Europe, the European Gymnastics Championships are held every two years and are open to all European FIG member federations.

At the national level, each country organizes its own system. In Slovenia, the **Gimnastična zveza Slovenije (GZS)** oversees national championships, regional competitions, and age-group events.
""")
    with col2:
        st.image("competitions 2.jpeg", use_container_width=True)

    st.markdown("---")

    # Section 2: competitions.jpg left, text right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image("competitions.jpg", use_container_width=True)
    with col2:
        st.markdown('<div style="color:#7C3AED;font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Competition Format</div>', unsafe_allow_html=True)
        st.markdown("### How a WAG Competition Works")
        st.markdown("""
A Women's Artistic Gymnastics competition follows a structured format across multiple rounds.

**Podium training:** Before the competition begins, teams have a familiarization period on the competition apparatus. This allows gymnasts to adjust to the specific equipment and environment.

**Qualifications:** All teams and individual gymnasts compete. Results determine who advances to the finals and serve as the team competition scores.

**Team final:** The top 8 teams advance. Each country fields up to 5 gymnasts, with 4 competing on each apparatus and the top 3 scores counting toward the team total.

**Individual all-around final:** The top 24 gymnasts (max. 2 per country) compete on all four apparatus. The combined score determines the all-around champion.

**Event finals:** The top 8 gymnasts per apparatus (max. 2 per country) compete again. Each apparatus has its own champion.

Gymnasts rotate in a fixed order: vault → uneven bars → balance beam → floor exercise.
""")

    st.markdown("---")

    # Section 3: text left, competitions 3.jpg right
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div style="color:#7C3AED;font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Slovenija</div>', unsafe_allow_html=True)
        st.markdown("### Tekmovalni sistem GZS")
        st.markdown("""
V Sloveniji so tekmovanja v ženski športni gimnastiki organizirana v dveh osnovnih programih: **stopenjskem programu** in **mednarodnem programu**.

**Stopenjski program** je namenjen predvsem dekletom, ki se z gimnastiko ukvarjajo rekreativno ali manj intenzivno. Tekmovanja potekajo po prilagojenem nacionalnem pravilniku GZS. Stopenjski program vključuje več stopenj (1–5 ter nadalje tudi višje stopnje), pri čemer se zahtevnost postopoma povečuje.

**Mednarodni program** je namenjen tekmovalkam, ki trenirajo gimnastiko bolj resno. V tem programu tekmovalke nastopajo po pravilih FIG, ki določa sestavo vaj, vrednotenje težavnosti (D-ocena) in izvedbe (E-ocena).

V praksi tekmovalke pogosto začnejo v stopenjskem programu in kasneje preidejo v mednarodni program.
""")
    with col2:
        st.image("competitions 3.jpg", use_container_width=True)


# ============================================================
# SEARCH PAGE
# ============================================================
def _expand_query(q: str) -> str:
    """Add gymnastics context to very short queries that lack sport-specific terms."""
    sport_terms = [
        "gymnast", "gymnastics", "gimnastik", "WAG", "MAG", "FIG",
        "floor", "beam", "vault", "bars", "score", "judge", "routine",
        "artistic", "rhythmic", "aerobic", "trampoline", "tumbling",
        "acrobatic", "parkour", "ocena", "tekmov", "sodnik", "vaja",
        "discipline", "disciplin", "competition", "tekmovanje",
    ]
    has_context = any(t.lower() in q.lower() for t in sport_terms)
    if len(q.split()) <= 3 and not has_context:
        return f"gymnastics {q}"
    return q


def show_search():
    st.markdown('<h2 class="section-header">Search / Iskanje</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#3B0764,#1E0A45);border-radius:8px;padding:28px 32px;margin-bottom:28px;">
        <div style="color:rgba(255,255,255,0.6);font-size:13px;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Knowledge Base</div>
        <div style="color:white;font-size:22px;line-height:1.6;">Ask a question in <strong style="color:white;">English or Slovenian</strong>. The search uses semantic similarity — natural language works well.</div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "Your question or topic:",
        value=st.session_state.search_query,
        placeholder="e.g.  How does the D score work?  /  Kako deluje ocenjevanje?"
    )
    st.session_state.search_query = query

    col1, col2 = st.columns([4, 2])
    with col1:
        lang_filter = st.radio(
            "Language",
            ["All", "English only", "Slovenian only"],
            horizontal=True
        )
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_search = st.button("Search", key="search_btn", use_container_width=True)

    run_search = run_search or st.session_state.auto_search
    st.session_state.auto_search = False

    if run_search:
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        expanded = _expand_query(query.strip())
        CANDIDATE_POOL = 30
        SCORE_THRESHOLD = 0.25

        # Primary: scored similarity search
        try:
            scored = vectorstore.similarity_search_with_relevance_scores(expanded, k=CANDIDATE_POOL)
            scored.sort(key=lambda x: x[1], reverse=True)
            raw_docs = [(doc, score) for doc, score in scored]
        except Exception:
            raw_docs = [(doc, 1.0) for doc in vectorstore.similarity_search(expanded, k=CANDIDATE_POOL)]

        # Secondary pass on original phrasing — merge new unique hits
        if expanded != query.strip():
            try:
                extra = vectorstore.similarity_search_with_relevance_scores(query.strip(), k=10)
                extra.sort(key=lambda x: x[1], reverse=True)
            except Exception:
                extra = [(doc, 1.0) for doc in vectorstore.similarity_search(query.strip(), k=10)]
            seen = {d.page_content for d, _ in raw_docs}
            for doc, score in extra:
                if doc.page_content not in seen:
                    raw_docs.append((doc, score))
                    seen.add(doc.page_content)

        # Filter by relevance threshold + language + deduplication
        filtered = []
        seen_content = set()
        for doc, score in raw_docs:
            if score < SCORE_THRESHOLD:
                continue
            content_key = doc.page_content[:120]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            lang = doc.metadata.get("language", "en")
            if lang_filter == "English only" and lang != "en":
                continue
            if lang_filter == "Slovenian only" and lang != "sl":
                continue
            filtered.append(doc)

        # Fallback: if threshold filtered everything out, return top 3
        if not filtered:
            fallback = []
            seen_content = set()
            for doc, _ in raw_docs:
                content_key = doc.page_content[:120]
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                lang = doc.metadata.get("language", "en")
                if lang_filter == "English only" and lang != "en":
                    continue
                if lang_filter == "Slovenian only" and lang != "sl":
                    continue
                fallback.append(doc)
                if len(fallback) >= 3:
                    break
            filtered = fallback

        if not filtered:
            st.info("No results found. Try a different query or change the language filter.")
            return

        st.markdown(f"<div style='color:#6B7280;font-size:15px;margin:16px 0 4px 0;'>{len(filtered)} result(s) for: <em>{query}</em></div>", unsafe_allow_html=True)

        for i, doc in enumerate(filtered):
            title      = doc.metadata.get("title", "")
            category   = doc.metadata.get("category", "")
            lang       = doc.metadata.get("language", "en")
            lang_label = "EN" if lang == "en" else "SL"
            content    = doc.page_content.replace("\n", "<br>")

            if i == 0:
                st.markdown('<div class="result-section-label" style="color:#F59E0B;">Best Match</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-card-best">
                    <div class="result-badge">Best Match</div>
                    <div class="result-title">{title}</div>
                    <div class="result-meta">{lang_label} &nbsp;·&nbsp; {category}</div>
                    <div class="result-content">{content}</div>
                </div>""", unsafe_allow_html=True)
            else:
                if i == 1:
                    st.markdown('<div class="result-section-label" style="color:#7C3AED;margin-top:32px;">Other Results</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">{title}</div>
                    <div class="result-meta">{lang_label} &nbsp;·&nbsp; {category}</div>
                    <div class="result-content">{content}</div>
                </div>""", unsafe_allow_html=True)


# ============================================================
# ABOUT PAGE
# ============================================================
def show_about():
    st.markdown(
        '<h2 class="section-header">About This Portal</h2>',
        unsafe_allow_html=True
    )

    st.markdown("""
### About

This portal is a beginner-friendly educational resource about gymnastics as a sport.
It covers all six gymnastics disciplines, competition formats, athlete training,
the WAG judging system, and gymnastics in Slovenia.

The judging system section is based exclusively on the **FIG Code of Points 2025–2028**,
the official rulebook of the Fédération Internationale de Gymnastique (FIG).

---

### How the Search Works

This application uses **Retrieval-Augmented Generation (RAG)**, a technique that
combines a document knowledge base with semantic search.

When you submit a query, the application:
1. Converts your query into a numerical vector (an embedding)
2. Searches the document database for the most semantically similar content
3. Returns the most relevant text passages

**Knowledge base:** 19 documents in English and Slovenian, split into chunks of
500 characters with 100-character overlap.

**Embedding model:** `paraphrase-multilingual-MiniLM-L12-v2` — supports both
English and Slovenian, so you can search in either language.

**Vector database:** ChromaDB (in-memory)

---

### How to Use This Portal

- **Home:** Browse by discipline. Click a card to navigate to the search page with
  the topic pre-filled.
- **Search:** Type any question or topic in English or Slovenian. Adjust the number
  of results and the language filter as needed.
- **About:** You are here.

---

### Sources

- FIG Code of Points (WAG) 2025–2028
- Fédération Internationale de Gymnastique — gymnastics.sport
- Gimnastična zveza Slovenije (GZS)
""")


# ============================================================
# ROUTING
# ============================================================
page_map = {
    "home": show_home,
    "about_sport": show_about_sport,
    "disciplines": show_disciplines,
    "disc_artistic": show_disc_artistic,
    "disc_rhythmic": show_disc_rhythmic,
    "disc_aerobic": show_disc_aerobic,
    "disc_trampoline": show_disc_trampoline,
    "disc_tumbling": show_disc_tumbling,
    "disc_acrobatic": show_disc_acrobatic,
    "disc_parkour": show_disc_parkour,
    "history": show_history,
    "judging": show_judging,
    "competitions": show_competitions,
    "search": show_search,
    "about_portal": show_about,
}
page_map.get(st.session_state.page, show_home)()
