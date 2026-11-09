// javis video: one entry per scene = one ElevenLabs take (Aom) + scenes/<id>.html. Every claim is from README.md (line in the comment).
// Markup inside text: [tag] = eleven_v4 delivery tag (never captioned) · ^ = timing anchor -> CSS --p2, --p3 ... (--p1 = voice start)
// {caption|spoken} = the caption shows the left side while the voice reads the right side. Brand words in SAY_AS get that treatment automatically.
// cap: real footage of a call (capture.mjs take, seconds from its recStart). Its own audio (caller + javis) plays all along, ducked under
// the narration. at = take second when the clip-only part starts: `hold` s after the narration, or `lead` s before it.
// from = take second at scene start instead. to = freeze the take there (its audio stops). Times picked from silencedetect on t2/call.webm.
export const SAY_AS = [
  ["javis.bcsmethai.com", "จาร์วิส ดอต บีซี เอสเอ็มอี ไทย ดอต คอม"], ["BC AI Account", "บีซี เอไอ แอคเคานต์"],
  ["javis", "จาร์วิส"], ["SQL", "เอสคิวแอล"], ["BC", "บีซี"], ["AI", "เอไอ"],
];

export const SCENES = [
  // problem framing only, no claim about javis
  { id: "s01", text: "[urgent] สายเข้าพร้อมกันหลายสาย ^ลูกค้าถามเรื่องเดิมซ้ำ ๆ ทั้งวัน ^ทีมรับสายก็มีอยู่ไม่กี่คน ^[curious] ถ้ามีเลขา ที่รับสายแรกแทน และตอบจากข้อมูลจริงได้ล่ะคะ" },
  // README:3, 6
  { id: "s02", text: "[excited] ขอแนะนำ จาร์วิส ค่ะ ^เลขาเสียงภาษาไทย คุยได้ทันที และพูดแทรกได้ เหมือนคุยโทรศัพท์ ^ตอบเรื่องโปรแกรมบัญชี บีซี เอไอ แอคเคานต์ จากคลังความรู้ ^โดยห้ามเดา และห้ามแต่งข้อมูล" },
  // README:300 (site starts the call, mic permission), voice_agent.py greet(): hello plays at once
  { id: "s03", text: "[warm] ลูกค้าแค่เปิดเว็บ ^สายก็เริ่มทันที ครั้งแรกเบราว์เซอร์จะขอสิทธิ์ใช้ไมค์ ^แล้ว จาร์วิส ก็ทักทายก่อนเลยค่ะ", cap: { take: "t2", at: 4.24 }, hold: 6.2 },
  // README:4, 25-28 (boss hands jobs to workers in parallel, team panel live)
  { id: "s04", text: "[curious] ลองถามด้วยภาษาพูดธรรมดาดูค่ะ ^หัวหน้าแจกงานให้ลูกทีม ไปค้นคลังความรู้ บีซี ^แผงทีมงานด้านซ้าย บอกสด ๆ ว่าใครกำลังทำอะไร", cap: { take: "t2", at: 24.0 }, hold: 6.7 },
  // README:6, 108, 120 (short speech, full detail on the panel, cards with their source)
  { id: "s05", text: "[thoughtful] คำตอบมาเป็นสองส่วนค่ะ ^พูดสรุปสั้น ๆ แบบเลขา ^ส่วนรายละเอียดเต็ม ขึ้นบนแผงกลางจอ ^พร้อมการ์ดบอกที่มาทางขวา", cap: { take: "t2", from: 31.5, to: 35.2 } },
  // README:53-56, 112 (calc does the maths, sample numbers are said to be made up)
  { id: "s06", text: "[excited] ขอตัวอย่างเป็นตารางก็ได้ค่ะ ^ตัวเลขคิดด้วยลูกทีมเครื่องคิดเลข ไม่ให้ เอไอ คิดเลขเอง ^และ จาร์วิส บอกทุกครั้งว่า ตัวเลขในตัวอย่าง เป็นตัวเลขสมมติ", cap: { take: "t2", at: 47.2 }, hold: 8.9 },
  // README:42 (SQL on screen, names from the search results only); in t2 the notes had no SQL for this and javis said so
  { id: "s07", text: "[thoughtful] ทีมซัพพอร์ต ขอคิวรี่ เอสคิวแอล ได้ด้วยค่ะ ^จาร์วิส เขียน เอสคิวแอล ขึ้นจอ จากชื่อตารางในผลค้นเท่านั้น ^ถ้าในคลังยังไม่มี ก็บอกตรง ๆ ว่าไม่มี ไม่แต่งคิวรี่ขึ้นมาเอง", cap: { take: "t2", from: 59.0, to: 69.6 } },
  // README:77-88 (number guard on every number, judge on every sentence, unsourced ones dropped silently)
  { id: "s08", text: "[serious] หัวใจของ จาร์วิส คือห้ามแต่งค่ะ ^ตัวเลขทุกตัว ทั้งในคำพูดและบนจอ ถูกโค้ดตรวจซ้ำกับแหล่งที่มา ^อีกชั้นหนึ่ง ตัวตรวจความหมาย อ่านทุกประโยค ^ประโยคไหนไม่มีที่มา ถูกตัดทิ้งทันที" },
  // README:3 (barge-in), voice_agent.py:808-829 (an off-topic jump the caller did not announce is asked about first)
  { id: "s09", text: "[playful] เห็นไหมคะ ผู้โทรพูดแทรกได้เลย ^จาร์วิส หยุดฟัง ^แล้วถามก่อนว่าจะเปลี่ยนเรื่องไหม ไม่เดาเอาเอง", cap: { take: "t2", at: 70.1, to: 84.5 }, lead: 12.7 },
  // README:36-47 (call-center expert persona, buyer: name, shop, phone one by one then read back; prices not in results -> ลุงจืด calls back)
  { id: "s10", text: "[excited] เอาไปช่วยงานอะไรได้บ้าง ^คอลเซ็นเตอร์ ให้ จาร์วิส รับสายแรก ตอบคำถามที่มีในคลังความรู้ ^ทีมขาย ลูกค้าที่สนใจ จาร์วิส ขอชื่อ ชื่อร้าน และเบอร์ทีละอย่าง แล้วอ่านทวนให้ฟัง ^ทีมซัพพอร์ต ตอบวิธีใช้โปรแกรม พร้อมขั้นตอนบนจอ ^ส่วนราคาหรือโปรโมชันที่ไม่มีในข้อมูล จาร์วิส ไม่รับปากเอง แต่บอกว่าจะให้ลุงจืดติดต่อกลับค่ะ" },
  // README:300, 312 (PUBLIC_CALLS=3 config.py:217), 18-19 (phone), 127 (typed chat), 36 (session memory)
  { id: "s11", text: "[warm] ยังมีอีกค่ะ ^รับได้สามสายพร้อมกัน ^โทรกลับจากเบราว์เซอร์เดิม จาร์วิส จำเรื่องที่เคยคุยไว้ได้ ^ใช้บนมือถือได้ ^และพิมพ์แชทแทนการพูดก็ได้ค่ะ" },
  { id: "s12", text: "[warm] อยากลองคุยกับ จาร์วิส เข้าไปที่ จาร์วิส ดอต บีซี เอสเอ็มอี ไทย ดอต คอม ^เปิดเว็บ แล้วพูดได้เลยค่ะ", hold: 2.5 },
];
