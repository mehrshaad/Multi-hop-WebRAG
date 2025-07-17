from RAG.__import__ import *


@lru_cache(maxsize=1)  # Cache the result to avoid repeated file reads
def load_wiki_data(filepath='./Data/2WikiMultihopQA/wiki_data-full-dict.json'):
    """Load and index the entire wiki data file once"""
    title_index = {}
    id_index = {}
    mention_index = defaultdict(list)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                title = normalize(obj.get("title", ""))
                obj_id = obj.get("id")

                if title:
                    title_index[title] = obj
                if obj_id:
                    id_index[obj_id] = obj

                mentions = obj.get("mentions", [])
                for m in mentions:
                    mention_text = m.get("text", "") if isinstance(
                        m, dict) else str(m)
                    norm_mention = normalize(mention_text)
                    if norm_mention:
                        mention_index[norm_mention].append(obj)

            except json.JSONDecodeError:
                continue

    return {
        'title_index': title_index,
        'id_index': id_index,
        'mention_index': mention_index
    }


def normalize(text):
    """Normalize text: lowercase, strip, remove diacritics."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text)
    return text.strip().lower()


def extract_titles_from_question(question):
    words = question.split()
    if not words:
        return []

    # New: Remove possessive 's from words before processing
    cleaned = [
        re.sub(r"'s\b", "", re.sub(r"[^\w\s\(\)\-\_]", "", w))
        for w in words[1:]
    ]

    titles = []
    i = 0
    while i < len(cleaned):
        w = cleaned[i]
        if (w and w[0].isupper()) or ("_" in w
                                      and all(part and part[0].isupper()
                                              for part in w.split("_"))):
            collector = [words[i + 1]]
            i += 1
            while i < len(cleaned):
                nw = cleaned[i]
                if ((nw and nw[0].isupper()) or "(" in nw or ")" in nw
                        or "-" in nw or "_" in nw):
                    collector.append(words[i + 1])
                    i += 1
                else:
                    break
            joined = " ".join(collector)
            if "(" in joined and ")" not in joined:
                while i < len(words):
                    collector.append(words[i + 1])
                    if ")" in words[i + 1]:
                        i += 1
                        break
                    i += 1
            # Remove possessive 's from the final title
            final_title = " ".join(collector).strip(" ,.?")
            final_title = re.sub(r"'s\b", "", final_title)
            titles.append(final_title)
        else:
            i += 1
    return titles


def extract_related_data(question, data_cache=None):
    """Modified version that uses pre-loaded data"""
    if data_cache is None:
        data_cache = load_wiki_data(
        )  # Fallback to loading if no cache provided

    wanted_titles = set(
        normalize(t) for t in extract_titles_from_question(question))
    collected = {}

    def add_entry(entry):
        if entry and entry.get("id") not in collected:
            collected[entry["id"]] = entry

    # Phase 1: Direct matches
    for t in wanted_titles:
        if t in data_cache['title_index']:
            add_entry(data_cache['title_index'][t])
        elif t in data_cache['mention_index']:
            add_entry(data_cache['mention_index'][t][0])

    # Phase 2: Resolve references
    for entry in list(collected.values()):
        for m in entry.get("mentions", []):
            ref_title = normalize(m.get("ref_url", ""))
            if ref_title in data_cache['title_index']:
                add_entry(data_cache['title_index'][ref_title])

            for rid in m.get("ref_id", []):
                if rid in data_cache['id_index']:
                    add_entry(data_cache['id_index'][rid])

    return [ent.get("sentences", []) for ent in collected.values()]


def get_related_sentences_hotpot(
        question_text,
        qa_file_path='./Data/HotpotQA/hotpot-dev.json',
        node_file_path='./Data/HotpotQA/hotpot-dev-context.json'):
    # Example usage
    # qa_path = 'hotpot_dev.json'
    # node_path = 'hotpot_dev-context.json'

    # Load QA dataset
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    # Load Node dataset
    with open(node_file_path, 'r', encoding='utf-8') as f:
        node_data = json.load(f)

    # Create quick title-based lookup for node dataset
    node_lookup = {item["title"]: item for item in node_data}

    supporting_titles = []
    for item in qa_data:
        if item.get("question") == question_text:
            supporting_titles = [
                fact[0] for fact in item.get("supporting_facts", [])
            ]
            break

    # Step 2: Gather all sentences from titles and mentions
    final_sentences = []
    visited_titles = set()  # To avoid duplicates

    def collect_sentences(title):
        if title in node_lookup and title not in visited_titles:
            visited_titles.add(title)
            node = node_lookup[title]
            final_sentences.extend(node.get("sentences", []))
            return node.get("mentions", [])
        return []

    mentions_to_follow = []
    for title in supporting_titles:
        mentions = collect_sentences(title)
        mentions_to_follow.extend(mentions)

    for mention in mentions_to_follow:
        collect_sentences(mention)

    return final_sentences
