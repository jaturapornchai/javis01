"""ห้ามแต่งตัวเลข: every number the secretary says or shows must appear in this turn's sources.

The prompt alone did not hold: with messy web results the boss invented gold prices, converted units and did its own
math on different runs. So code checks it: numbers in digits, Thai digits or Thai number words are compared with the
numbers in the question, the clock, earlier turns and every worker result; a sentence or screen line with any other
number is cut. Self-test: python grounding.py
"""
import difflib
import re

from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize
from pythainlp.util import text_to_num, thai_digit_to_arabic_digit, thaiword_to_num

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# a rate must be a rate in the sources too ("ร้อยละสิบ" passed on a 10 that was a row count); a lone 0-9 is judged
# only as a rate or an amount (VAT 7%, 5 บาท, ภายใน 7 วัน): bare ones are too common (ชู 2 นิ้ว, ข้อ 3)
RATE = re.compile(rf"%\s*({NUM.pattern})|({NUM.pattern})\s*(?:%|เปอร์เซ็นต์)|(?<![\d.,])(\d)\s*(?:บาท|ดอลลาร์|วัน|สัปดาห์|เดือน|ปี)")
UNIT = {"สิบ": "10", "ร้อย": "100", "พัน": "1000", "หมื่น": "10000", "แสน": "100000", "ล้าน": "1000000"}
DIGIT_WORDS = "ศูนย์ หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า".split()
# digits read one by one (a phone number: ศูนย์แปดหนึ่ง...) made pythainlp raise and the whole turn died;
# a run not followed by a unit word is plain digits (สองศูนย์สามสิบ stays 20 + 30)
DIGIT_RUN = re.compile(rf"(?:(?:{'|'.join(DIGIT_WORDS)})\s*){{2,}}(?!{'|'.join(UNIT)})")
STOP = thai_stopwords()
CUT_SHOW = "(ตัดบรรทัดที่ไม่มีในแหล่งข้อมูลออกแล้ว)"


def words_to_digits(text):
    """pythainlp's number-word pass. It raises on some real text (a long bcbrain search result did, and the whole
    turn died), so then piece by piece; a piece it still can't read keeps only the digits it already has."""
    try:
        return text_to_num(text + " ครับ")  # pythainlp drops a number word ending the text
    except ValueError:
        out = []
        for piece in text.split():
            try:
                out += [*text_to_num(piece + " ครับ"), " "]
            except ValueError:
                out += [piece, " "]
        return out


def numbers(text):
    """Every number as plain digits; a lone digit only as a rate/amount, marked "7#"."""
    toks = []
    text = re.sub(r"ร้อย\s*ละ", "%", thai_digit_to_arabic_digit(text))  # ร้อยละ N = % N, so its ร้อย is no 100
    text = DIGIT_RUN.sub(lambda m: "".join(str(DIGIT_WORDS.index(w)) for w in re.findall("|".join(DIGIT_WORDS), m[0])), text)
    for tok in words_to_digits(text):
        tok = "0" if tok == "ศูนย์" else UNIT.get(tok, tok)  # and leaves a lone ศูนย์ / สิบ / พัน as a word
        if toks and tok[:1].isdigit() and toks[-1][-1:].isdigit():
            toks[-1] += tok  # digits read one by one are one number: สองศูนย์สามสิบ = 2030, a phone number
        else:
            toks.append(tok)
    plain = " ".join(toks)
    found = {n.replace(",", "") for n in NUM.findall(plain)}
    return {n for n in found if len(n) > 1} | {(a or b or c).replace(",", "") + "#" for a, b, c in RATE.findall(plain)}


def unsourced(text, known):
    """Numbers in text that no source has. A rounded 3,931 for a sourced 3,931.43 still counts as sourced."""
    # ponytail: exact match only; converted years (2026 -> 2569) or sums the boss works out are cut too, by design
    return {n for n in numbers(text) if n not in known and not any(k.split(".")[0] == n for k in known)}


def note_sections(note, question):
    """The '## ' sections of a full note whose heading has a word of the question (ยกเลิกยังไง -> ## ยกเลิก/แก้ไข)."""
    # ponytail: heading match only; a question whose words are in no heading gets nothing extra (snippet only)
    words = [w for w in word_tokenize(question, keep_whitespace=False) if len(w) > 2 and w not in STOP]
    return "\n".join("## " + p for p in note.split("\n## ")[1:] if any(w in p.split("\n", 1)[0] for w in words))


def is_number(word):
    try:
        thaiword_to_num(word)
        return True
    except ValueError:
        return False


# spoken readings of what sources abbreviate: the boss must say "4 พ.ย." as "สี่พฤศจิกายน"; an era name adds
# nothing to a year (numbers() checks the year), so it is always fine
READINGS = dict(zip("ม.ค. ก.พ. มี.ค. เม.ย. พ.ค. มิ.ย. ก.ค. ส.ค. ก.ย. ต.ค. พ.ย. ธ.ค.".split(), (
    "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม").split()))


# how Thai speakers read Latin letters (ลุงจืด 2026-11-04 "ระบบพีโอเอส = ระบบ POS"): the boss must say a source's
# POS as พีโอเอส, and Whisper writes a caller's POS that way too
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "เอ บี ซี ดี อี เอฟ จี เอช ไอ เจ เค แอล เอ็ม เอ็น โอ พี คิว อาร์ เอส ที ยู วี "
                   "ดับเบิลยู เอ็กซ์ วาย แซด".split()))
LETTER = "|".join(sorted(LETTERS.values(), key=len, reverse=True))  # longest first: เอ็ม before เอ
# two or more letter names in a row, none running on into a vowel or tone mark (ดีที่สุด is no "DT", เอา no "A")
SPELLED = re.compile(rf"(?:(?:{LETTER})(?![ะ-ฺๅ็-๎])){{2,}}")


def spelled(text):  # "ระบบ POS" -> ["พีโอเอส"]
    return ["".join(LETTERS[c] for c in a) for a in re.findall(r"(?<![A-Za-z])[A-Z]{2,6}(?![A-Za-z])", text)]


def acronyms(text):  # "ระบบพีโอเอส" -> ["POS"]; ceiling: a letter-like word glued on joins it (พีโอเอสดี -> POSD)
    back = {v: k for k, v in LETTERS.items()}
    return ["".join(back[t] for t in re.findall(LETTER, m)) for m in SPELLED.findall(text)]


def is_echo(heard, spoken):
    """Most words the mic heard are words the secretary was saying meanwhile: its own voice back from the speaker."""
    # ponytail: a caller who repeats the bot's own words over it is taken as echo too; said again once it stops
    words = [w for w in word_tokenize(heard, keep_whitespace=False) if re.search(r"\w", w)]
    low = spoken.replace(" ", "")
    return bool(words) and sum(w in low for w in words) >= 0.6 * len(words)


def new_words(piece, sources):
    """Words of `piece` that `sources` never use (number words aside: numbers() checks those). Only a piece with some
    goes to the meaning judge; one made of source words passes as is."""
    # ponytail: source words put together into a new claim pass ("ยกเลิกผ่านระบบบัญชีที่ใช้อยู่"); judging every
    # phrase instead costs ~1 s each and the 8B judge says yes to almost all of them anyway
    low = " ".join([sources, "พุทธศักราช คริสต์ศักราช", *(r for a, r in READINGS.items() if a in sources), *spelled(sources)]).lower()
    seen = lambda w: len(w) > 1 and (w in STOP or is_number(w) or w.lower() in low)
    def known(w):  # a source word, or one glued to a stopword by the tokenizer: ใช้กับ = ใช้ + กับ
        return seen(w) or any((w.startswith(t) and seen(w[len(t):])) or (w.endswith(t) and seen(w[:-len(t)]))
                              for t in STOP if len(t) < len(w))
    new = [w for w in word_tokenize(piece, keep_whitespace=False) if len(w) > 1 and re.search(r"\w", w) and not known(w)]
    # a near spelling of source text is that word: the 8B judge called the boss's ทันตกรรม new_info three times out of
    # three though the caller's own words (Whisper) said ทันตะกรรม, and cut the clinic's name. The tokenizer splits a
    # misspelling, so compare with raw source windows. ponytail: 5+ letters at 0.85 (ปิด/เปิด stay apart); ceiling: a
    # different long word one letter off a source word passes unjudged; ~0.1 s a word on 12 KB of sources
    near = lambda w: len(w) >= 5 and difflib.get_close_matches(
        w.lower(), {low[i:i + n] for n in (len(w) - 1, len(w), len(w) + 1) for i in range(len(low) - n + 1)}, 1, 0.85)
    return [w for w in new if not near(w)]


if __name__ == "__main__":
    known = numbers("ราคาทองวันนี้ 3,931.43 ดอลลาร์ ทองคำแท่ง ขายออก 62,500 บาท อัปเดต ๔ พ.ย. ๒๕๖๙ (ระบบ: เวลา 08:19 น.)")
    assert known == {"3931.43", "62500", "2569", "08", "19"}, known
    tax = numbers("ภาษีมูลค่าเพิ่มร้อยละ 7 ของยอดขาย")
    assert tax == {"7#"}, tax
    assert not unsourced("คิดภาษีเจ็ดเปอร์เซ็นต์ครับ", tax) and not unsourced("ร้อยละเจ็ด", tax)
    assert unsourced("ค่าเช่าหัก 3%", tax) == {"3#"} and unsourced("หักร้อยละสาม", tax) == {"3#"}  # invented tax rates
    assert unsourced("ภาษีหัก ณ ที่จ่ายค่าเช่าร้อยละสิบครับ", tax) == {"10", "10#"}  # lone สิบ, the real miss
    assert unsourced("หัก 10%", numbers("ตาราง 10 แถว")) == {"10#"}  # a 10 that is no rate does not source a rate
    assert not unsourced("หักร้อยละสิบ ค่าขนส่ง 1%", numbers("อัตรา 10% และ 1 เปอร์เซ็นต์"))
    assert numbers("ร้อยบาท สิบโมง") == {"100", "10"} and numbers("ห้าสิบบาท") == {"50"}
    assert not unsourced("ชูสองนิ้ว เลือกข้อสาม", tax)
    assert not unsourced("ฟุตบอลโลกปีสองศูนย์สามสิบยังไม่แข่ง", numbers("ใครชนะฟุตบอลโลกปี 2030"))
    assert not unsourced("ทองคำแท่งขายออกหกหมื่นสองพันห้าร้อยบาทครับ", known)  # spoken words = sourced digits
    assert not unsourced("ราว 3,931 ดอลลาร์ ชู 2 นิ้ว", known)  # rounding of a sourced number, single digits
    assert not unsourced("ตอนนี้แปดโมงสิบเก้านาที ปีสองพันห้าร้อยหกสิบเก้า", known)
    assert unsourced("ทองรูปพรรณขายออก 63,800 บาท", known) == {"63800"}  # invented price
    assert unsourced("ราคา ๒,๙๘๐ บาท", known) == {"2980"}  # invented, in Thai digits
    assert unsourced("ประมาณหนึ่งแสนสี่หมื่นเจ็ดพันบาทต่อกิโล", known) == {"147000"}  # its own math, in words
    note = "# ขายสินค้า\nใช้ขาย\n## ใช้เมื่อไร\nขายเชื่อ\n## ยกเลิก/แก้ไข\n- ยกเลิก 44 ด้วย flag 45\n## ข้อควรระวัง\nวงเงิน"
    assert note_sections(note, "ใบกำกับภาษีขายยกเลิกยังไง") == "## ยกเลิก/แก้ไข\n- ยกเลิก 44 ด้วย flag 45"
    assert note_sections(note, "ร้านกาแฟเปิดกี่โมง") == ""  # off-topic top hit adds nothing
    shop = "ร้านกาแฟลุงจืด เปิดทุกวัน เวลาเจ็ดโมงเช้าถึงหกโมงเย็น ปิดทุกวันพุธ"
    assert not new_words("ร้านกาแฟลุงจืดเปิดเจ็ดโมงเช้าถึงหกโมงเย็น และปิดทุกวันพุธครับ", shop)
    assert new_words("มีที่จอดรถฟรีด้วยครับ", shop) == ["ที่จอดรถ", "ฟรี"] and new_words("ปิดทุกวันจันทร์", shop) == ["จันทร์"]
    assert not new_words("| ปิด | **ทุกวันพุธ** |", shop)  # markup and number words are no new words
    assert not new_words("ใช้กับสินค้าทุกตัว", "ใช้ถัวเฉลี่ยกับสินค้าทุกตัว") and new_words("การจัดส่ง", shop) == ["การจัดส่ง"]
    assert not new_words("วันพุธที่สี่พฤศจิกายน พุทธศักราชสองพันห้าร้อยหกสิบเก้า", "พุธ 4 พ.ย. 2569")
    assert not new_words("คุณมณีบอกชื่อว่าคลินิกมณีทันตกรรมครับ", "ดิฉันมณีคลินิกมณีทันตะกรรมขอให้ติดต่อกลับ บอก ชื่อ ว่า")
    assert new_words("คลินิกศัลยกรรม", "คลินิกมณีทันตะกรรม") == ["ศัลยกรรม"] and new_words("คุณสมชัย", "คุณสมชาย") == ["ชัย"]  # another name
    assert numbers("ใบที่ 12 ราคาห้าสิบหกสิบบาท") == {"12"}  # pythainlp raises on this range: the rest still counts
    phone = numbers("ร้านสมหญิงวัสดุ เบอร์ศูนย์แปดหนึ่ง หนึ่งสองสาม สี่ห้าหกเจ็ดค่ะ")
    assert phone == {"0811234567"}, phone
    assert not unsourced("เบอร์ศูนย์แปดหนึ่ง หนึ่งสองสาม สี่ห้าหกเจ็ด นะครับ", phone)  # read back: no crash, sourced
    assert unsourced("เบอร์ศูนย์แปดหนึ่ง หนึ่งสองสาม สี่ห้าหกแปด", phone) == {"0811234568"}  # misheard digit
    assert unsourced("แล้วออกใบใหม่ภายในเจ็ดวันครับ", numbers("ยกเลิก 44 ด้วย flag 45")) == {"7#"}  # a made-up time limit
    # Thai readings of Latin letters, both ways; ดีที่สุด / เอาไว้ / อีกที are words, not letters
    assert acronyms("ระบบพีโอเอส") == ["POS"] and acronyms("เอาทีวีไปไอที") == ["TV", "IT"] and not acronyms("ดีที่สุด เอาไว้ อีกที")
    assert not new_words("ระบบพีโอเอสครับ", "เมนูระบบ POS ขายหน้าร้าน")  # the judge cut such a source term before
    assert is_echo("ผมเป็นเลขาของ ลุงจืด", "สวัสดีครับ ผมเป็นเลขาของลุงจืดครับ") and not is_echo("ราคาทองวันนี้เท่าไหร่", "สวัสดีครับ")
    print("grounding self-test OK")
