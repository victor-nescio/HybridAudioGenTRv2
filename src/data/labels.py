"""
BBC Ses Arşivi (F: diski) için 26 sınıflı Türkçe taksonomi.

Öncelikli eşleşme sistemi: _KURALLAR listesinde ilk eşleşen kural kazanır.
AudioLDM için metin prompt'ları, LGBM/AST için dengeli sınıf yapısı.
"""

# ---------------------------------------------------------------------------
# 26 sınıf: {klasör_adı: (Türkçe_görünen_ad, AudioLDM_prompt)}
# ---------------------------------------------------------------------------
SINIFLAR: dict[str, tuple[str, str]] = {
    "ucak_sesleri":        ("Uçak Sesleri",           "airplane flying overhead"),
    "tren_sesleri":        ("Tren Sesleri",            "steam train passing through station"),
    "kara_tasitlari":      ("Kara Taşıtları",          "cars and buses in city traffic"),
    "gemi_tekne_sesleri":  ("Gemi ve Tekne Sesleri",   "ship horn on ocean"),
    "at_sesleri":          ("At Sesleri",              "horses galloping on gravel road"),
    "kopek_sesleri":       ("Köpek Sesleri",           "dog barking loudly"),
    "kus_sesleri":         ("Kuş Sesleri",             "birds singing at dawn"),
    "diger_hayvanlar":     ("Diğer Hayvan Sesleri",    "farm animals on a rural farm"),
    "hava_olaylari":       ("Hava Olayları",           "thunderstorm with heavy rain"),
    "su_sesleri":          ("Su Sesleri",              "river flowing over rocks"),
    "doga_atmosferi":      ("Doğa Atmosferi",          "peaceful countryside ambience"),
    "sehir_atmosferi":     ("Şehir Atmosferi",         "busy city street ambience"),
    "kalabalik_sesleri":   ("Kalabalık Sesleri",       "large crowd cheering at stadium"),
    "cocuk_bebek_sesleri": ("Çocuk ve Bebek Sesleri",  "children playing in schoolyard"),
    "silah_patlama":       ("Silah ve Patlama",        "explosions and gunfire in battle"),
    "acil_servis":         ("Acil Servis Sesleri",     "ambulance siren in city"),
    "endustri_fabrika":    ("Endüstri ve Fabrika",     "heavy machinery in factory"),
    "insaat_sesleri":      ("İnşaat Sesleri",          "construction site with drilling"),
    "ev_ici_sesler":       ("Ev İçi Sesler",           "kitchen appliances running"),
    "kapi_sesleri":        ("Kapı Sesleri",            "wooden door creaking open and shut"),
    "saat_can_sesleri":    ("Saat ve Çan Sesleri",     "church bells ringing at noon"),
    "ayak_sesleri":        ("Ayak Sesleri",            "footsteps walking on gravel"),
    "ofis_iletisim":       ("Ofis ve İletişim",        "typewriter and telephone in office"),
    "tarim_ciftlik":       ("Tarım ve Çiftlik",        "tractor working in farm field"),
    "spor_eglence":        ("Spor ve Eğlence",         "crowd noise at football match"),
    "dini_mekan":          ("Dini Mekan Sesleri",      "church atmosphere with organ music"),
}

# ---------------------------------------------------------------------------
# Öncelikli eşleşme kuralları (ilk eşleşen kural kazanır).
# Anahtar kelimeler klasör/dosya adına küçük harfle substring olarak uygulanır.
# ---------------------------------------------------------------------------
_KURALLAR: list[tuple[str, list[str]]] = [

    # ── Köpekler ──────────────────────────────────────────────────────────
    ("kopek_sesleri", [
        "airedale", "bulldog", "collie", "retriever", "spaniel",
        "wolfhound", "beagle", "huskie", "huskies", "samoyed",
        "maltese terrier", "mongrel", "cairn terrier",
        "wire-haired fox terrier", "west highland terrier",
        "dog ", "dog,", "dogs_", "dogs,", "dogs ", "dogs\n",
        " dogs", "dogs", "puppy",
    ]),

    # ── Atlar ─────────────────────────────────────────────────────────────
    ("at_sesleri", [
        "horse", "horses", "pony", "equestrian",
        "horsedrawn", "horse-drawn", "hansom",
        "mail coach", "coach & four", "one-horse",
        "two-horse", "coaching inn", "sleigh",
    ]),

    # ── Kuşlar ────────────────────────────────────────────────────────────
    ("kus_sesleri", [
        "bird", "birds", "pigeon", "parrot", "budgerigar",
        "chicken", "chickens", "pheasant", "poultry",
        "dawn chorus", "farm birds", "british birds",
        "owl imitation", "cicadas",
    ]),

    # ── Diğer hayvanlar ───────────────────────────────────────────────────
    ("diger_hayvanlar", [
        "cats", "cat ", "kittens",
        "sheep", "lamb", "lambs", "goat", "goats",
        "cattle", " pig", "pigs", "donkey", "donkeys",
        "livestock", "farm animals", "animals",
        "mice", "rat ", "rats", "opossum", "polar bear",
        "reindeer", "dragon", "pet shop",
    ]),

    # ── Uçak ──────────────────────────────────────────────────────────────
    ("ucak_sesleri", [
        "aircraft", "helicopter", "boeing", "concorde",
        "airfield", "airport", "aeroplane",
        "de havilland", "tiger moth", "piper ",
        "cessna", "vickers vanguard", "vickers viscount",
        "bristol britannia", "fairey", "shackleton",
        "flying fortress", "junkers", "messerschmitt",
        "gloster", "starfighter", "vampire ",
        "mosquito fighter", "spitfire", "lancaster",
        "world war ii aircraft", "world war i aircraft",
        "military aerodrome", "vulcan bomber",
        "brantley", "sikorski", "westland lynx",
        "whirlwind helicopter", "sea king helicopter",
        "wessex ii", "hercules, aircraft",
        "vc10", "rocket motor", "space ship",
        "space travel", "u.s. space", "farnborough",
        "bristol prop", "lightning twin-jet",
        "r.a.f.", "prop-jet", "airliner", "boac",
    ]),

    # ── Tren ve tramvay ───────────────────────────────────────────────────
    ("tren_sesleri", [
        "steam train", "diesel train", "electric train",
        "train ", "trains", "trams", "tram", "tram,", "tram ",
        "railway", "underground", "metro",
        "euston", "victoria station", "rail terminus",
        "eurostar", "narrow-gauge", "model railway",
        "signal box", "steam locomotive", "diesel railcar",
        "diesel electric train", "southern region",
        "british rail", "steam goods",
        "age of steam", "the age of steam",
    ]),

    # ── Gemi ve tekne ─────────────────────────────────────────────────────
    ("gemi_tekne_sesleri", [
        "ship ", "ships", "boat ", "boats", "vessel",
        "trawler", "tug ", "tug,", "ferry", "lifeboat",
        "submarine", "hovercraft", "hobercraft", "dinghy",
        "sailing", "yacht", "barge", "paddle steamer",
        "canal", "canoe", "rowing", "harbour",
        "wharf", "motor launch", "beam fishing",
        "container ship", "cabin cruiser", "tug",
        "steam launch", "marine-type",
    ]),

    # ── Kara taşıtları ────────────────────────────────────────────────────
    ("kara_tasitlari", [
        "motor car", "motor cycle", "motorcycle",
        " bus ", "buses", "bus_", "bus,", "bus horn",
        "truck", "lorry", "van ", " van_",
        "taxi", "motor traffic", "traffic",
        "motorway", "ford ", "renault ",
        "mini ", "austin ", "triumph tr",
        "veteran car", "vintage car", "vintage motor",
        "motor horn", "motor racing", "formula one",
        "formula 1", "stock car",
        "land rover", "motor scooter", "scooter",
        "ambulance (", "police car",
        "cars", "car crash", "car park",
        "skids", "peugeot", "wolseley",
        "rover saloon", "bicycle", "bicycles",
        "vintage alldays", "vintage benz",
        "vintage mercedes", "vintage wolseley",
        "bus depot", "coach horn", "petrol station", "road tunnel",
    ]),

    # ── Hava olayları ─────────────────────────────────────────────────────
    ("hava_olaylari", [
        "thunder", " rain", "rain ", "wind",
        "storm", "hail", "fog",
        "sandstorm", "avalanche", "thunderstorm",
        "blizzard", "gale", "volcanic",
    ]),

    # ── Su sesleri ────────────────────────────────────────────────────────
    ("su_sesleri", [
        "water", "sea ", "seas", "river",
        "stream", "waterfall", "seawash",
        "wave", "weir", "fountain",
        "splash", "swimming", "victoria falls",
        "rushing water", "mountain stream",
        "punting", "diving into water",
        "person swimming", "bubbling liquid",
        "scuba diving",
    ]),

    # ── Acil servis ───────────────────────────────────────────────────────
    ("acil_servis", [
        "fire engine", "fire service", "fire brigade",
        "fire appliance", "police car",
        "999", "fire_ ambulence",
        "ambulances", "fire engines",
        "german emergency", "hospital",
        "pulse monitor", "police ar_",
        "volvo 244 police", "dental", "firefighting",
    ]),

    # ── Doğa atmosferi ────────────────────────────────────────────────────
    ("doga_atmosferi", [
        "woodland", "countryside", "rural",
        "garden", " forest", "jungle", "tropical",
        "dawn chorus", "country scene",
        "summer atmosphere", "coastal",
        "highland", "mountain ",
        "amazon", "patagonia",
        "new zealand", "africa - natural",
        "natural world", "wild places",
        "autumn atmosphere", "coastal garden",
        "devon hedgerow", "undergrowth",
        "natural atmosphere", "norfolk",
        "greenland", "exterior atmosphere",
        "atmospheres", "hampshire farm",
        "bonfire", "fire", "quiet park", "swaziland", "wooden creek",
    ]),

    # ── Çocuk ve bebek ────────────────────────────────────────────────────
    ("cocuk_bebek_sesleri", [
        "baby", "babies", "children", " child",
        "infant", "schoolchildren", "toddler",
        "boys_ school", "boys_ comp", "boys_ sec",
        "girls_ school", "girls_ gram", "girls_ sec",
        "young children", "small children",
        "5-10 year", "30 children",
        "mixed school", "school dining", "toys",
    ]),

    # ── Kalabalık ─────────────────────────────────────────────────────────
    ("kalabalik_sesleri", [
        "crowd", "crowds", "audience",
        "applause", "demonstration",
        "protest", "spectator", "cheering",
        "football crowd", "rugby crowd",
        "large outdoor", "exhibition crowd",
        "expectant crowd", "laughter",
        "chatter", "theatre", "concert hall",
        "royal albert", "royal festival",
        "interiors - concert", "interiors - theatre",
        "playhouse", "people in", "screaming", "cocktail",
        "canteen", "teenage",
    ]),

    # ── Şehir atmosferi ───────────────────────────────────────────────────
    ("sehir_atmosferi", [
        "london", "city ", "urban",
        "street ", "suburb", " town",
        "village ", "market",
        "foreign", "european cities",
        "paris", "rome", "new york",
        "istanbul", "india and nepal",
        "general background", "skyline",
        "in a city", "provincial town",
        "america", "china", "greece",
        "hungary", "spain", "france",
        "england", "rural south america",
        "urban south america", "africa 1", "africa 2",
        "india, pakistan", "sydney", "scandinavia",
        "romania", "poland", "germany",
        "italy", "belgium", "egypt",
        "ethiopia", "botswana",
        "restaurant", "pub ", "hotel ",
        "shops", "snack bar", "museum",
        "gallery", "airport",
        "interiors - restaur", "interiors - pubs",
        "interiors - shops", "interiors - various",
        "interior background", "public house",
        "betting shop", "gambling",
        "dance hall",
        "cambridge", "beirut", "lebanon", "czechoslovakia", "holland",
        "hotel", "post office", "public bar", "shopping", "newsagent",
        "confectioner", "gasthof", "fish & chip", "butcher",
        "busy bank", "reception hall", "refuse collection",
        "tea tent", "club bar", "clubs and", "hairdressing",
    ]),

    # ── Silah ve patlama ──────────────────────────────────────────────────
    ("silah_patlama", [
        "gunfire", "gun ", "guns_", "explosion",
        "bomb", "battle", "warfare",
        "artillery", "cannon", "tank ",
        "world war", "period battle",
        "napoleonic", "small arms",
        "heavy explosion", "air battle",
        "air raid", "naval bombardment",
        "v1 flying bomb", "v2 rocket",
        "antique firearm", "bang_",
        "blows & crash", "shooting",
        "military ", "army",
        "grenades", "warfare_ vietnam",
        "shotgun", "armoured car",
        "bugle call", "saracen",
        "crash 1", "crash 2", "crashes",
        "metal crash", "glass crash",
        "wooden crash", "metal squeak",
        "skid", "plymouth fire",
    ]),

    # ── Dini mekan ────────────────────────────────────────────────────────
    ("dini_mekan", [
        "church atmosphere", "cathedral atmosphere",
        "monastery", "religion in britain",
        "ecclesiastical", "coventry cathedral",
        "st. paul", "abbey ",
        "chapel ", "worship", "churchyard",
        "chuchyard", "monasteri",
    ]),

    # ── Kapı sesleri ──────────────────────────────────────────────────────
    ("kapi_sesleri", [
        "door ", "door,", "doors", " gate ", " gates",
        "drawer", "prison door", "cell door",
        "cupboard door", "garage door",
        "metal door", "steel door",
        "heavy door", "creaking door",
        "sliding door", "locks, knocker",
        "barn door", "iron gate",
        "knocks on a door", "knocked on",
    ]),

    # ── Saat ve çan sesleri ───────────────────────────────────────────────
    ("saat_can_sesleri", [
        " clock", "clocks",
        "bells", "bell ",
        "chime", "chimes",
        "ticking", "striking",
        "big ben", " bong",
        "church bell", "cathedral bell",
        "handbells", "goat bells",
        "sleigh bells", "bicycle bell",
        "bracket clock", "domestic clock",
        "bells ringing", "gong", "cymbal",
    ]),

    # ── Ev içi sesler ─────────────────────────────────────────────────────
    ("ev_ici_sesler", [
        "household", "kitchen",
        "washing machine", "vacuum cleaner",
        "refrigerator", "bathroom",
        "launderette", "domestic radio",
        "sewing machine", "house door",
        "home ", " home,",
        "kettle", "tap drip", "tap dribbling",
        "lifts", "lift,", "lift ",
        "coughing", "breathing", "squeaks", "washing line",
    ]),

    # ── Ayak sesleri ──────────────────────────────────────────────────────
    ("ayak_sesleri", [
        "footstep", "footsteps",
        "feet in", "clog",
        "walking on", "running on",
        "on gravel", "on pavement",
        "on stone", "on wood",
        "on hard surface", "on cobble",
        "on country road", "on road",
        "squelching footstep",
        "fast footsteps", "slow footsteps",
        "men walking", "woman walking",
        "men entering room", "men leaving room", "men running",
    ]),

    # ── İnşaat sesleri ────────────────────────────────────────────────────
    ("insaat_sesleri", [
        "construction", "building work",
        "demolition", "pneumatic drill",
        "bricklaying", "ready-mixed cement",
        "building site", "building under",
        "bridge building", "motorway construction",
        "road re-surfacing", "building construction",
        "d.i.y", "excavator", "crane",
        "tree-felling", "hammering",
        "wall being pulled", "carpebter", "drawbridge",
    ]),

    # ── Endüstri ve fabrika ───────────────────────────────────────────────
    ("endustri_fabrika", [
        "factory", "industry", "industrial",
        "manufacturing", " mill ", "mining",
        "gasworks", "power station",
        "oil rig", "oil refinery",
        "lathe", "forge", "machine shop",
        "machinery", "engine ", "engines",
        "drilling for oil", "atomic power",
        "heavy engineering", "light industry",
        "sheet metal", "timber yard",
        "brewery", "brickworks", "pottery",
        "papermaking", "textile",
        "coal mining", "granite quarry",
        "sewage pump", "water pumping",
        "dynamo", "carpenter", "carpentry",
        "grinding", "circular saw", "band saw",
        "timber", "waste disposal",
        "pulleys", "chemical",
        "glasswork", "glass crash",
        "blacksmith", "metalwork",
        "sawmill", "wood crushing",
        "wood creaking", "wood crushed",
        "wood splintering", "wooden structure",
        "large wooden", "timber crushed",
        "wood slowly", "cottage industr",
        "2-stroke", "mechanical digger",
        "escaping steam", "barrels rolling",
        "loose stones", "loose dirt",
        "dressmaking", "shoe repair",
        "factories", "warehouse", "scrap yard", "oil exploration",
        "oil & petrol", "trouser", "electrical radiation",
        "brewing", "mills & grain",
    ]),

    # ── Ofis ve iletişim ──────────────────────────────────────────────────
    ("ofis_iletisim", [
        "typewriter", "telephone",
        "computer", "office ",
        "printer", "telegraph",
        "morse", "radio ",
        "communication", "switchboard",
        "cash register", "electronic sound",
        "electrical sound", "electronic alarm",
        "sine wave", "warble frequency",
        "interference", "static",
        "slot machine", "electronic",
        "electrical-", "cameras",
        "more phones", "newspapers",
        "printing", "telemetr",
        "bbc news sound", "national daily",
        "cine projector", "talking toy",
        "offices", "meeting", "library", "lloyd", "underwriting",
        "ticket machine",
    ]),

    # ── Tarım ve çiftlik ──────────────────────────────────────────────────
    ("tarim_ciftlik", [
        "farm machine", "farm machin",
        "tractor", "combine harvester",
        "agriculture", "dairy farm",
        "farming", "barn ", "field ",
        "ferguson tractor", "perkins-fordson",
        "corn mill", "mowing",
        "dairi", "machnery", "mower", "motor mower",
    ]),

    # ── Spor ve eğlence ───────────────────────────────────────────────────
    ("spor_eglence", [
        "football", "cricket",
        "tennis", "boxing", "athletics",
        " ski", "cycling",
        "fairground", "circus",
        "gymnasium", " match",
        "tournament", "bowling",
        "racing", "wimbledon",
        "swimming pool", "ice rink",
        "skateboard", "archery",
        "fencing", "adventure",
        "motor racing", "greyhound",
        "billiard", "basketball",
        "badminton", "handball",
        "sport", "leisure",
        "fox hunting", "caving",
        "fitness", "cup final", "f.a. cup",
        "roller skat", "firework",
        "wrestling", "procession",
        "fun fair", "gaming", "pool game", "turnstile", "basket ball",
    ]),
]


def siniflandir(isim: str) -> str | None:
    """
    Klasör veya dosya adını alır, ilgili Türkçe sınıf klasör adını döndürür.
    Hiçbir kural eşleşmezse None döner.
    """
    isim_kucuk = isim.lower()
    for sinif, anahtarlar in _KURALLAR:
        for anahtar in anahtarlar:
            if anahtar in isim_kucuk:
                return sinif
    return None


def turkce_ad(sinif: str) -> str:
    return SINIFLAR.get(sinif, (sinif,))[0]


def prompt_al(sinif: str) -> str:
    return SINIFLAR.get(sinif, ("", sinif))[1]
