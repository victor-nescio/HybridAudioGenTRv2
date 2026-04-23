import re

# BBC klasör adından prefix kaldırıldıktan sonra şehir/kentsel sayılan kategoriler
_SEHIR_KENTSEL = {
    "European Soundscapes",
    "America",
    "China",
    "Greece",
    "Hungary",
    "Spain",
    "France",
    "England",
    "Istanbul",
    "India & Nepal-City Life",
    "Urban South America",
    "Africa_ The Human World",
}

# Kırsal/doğal bölge kategorileri — "City Sounds" değil, "Rural Sounds" olarak ayrılır
_KIRSAL_DOGAL = {
    "Africa - Natural World",
    "India Pakistan Nepal-Countrysid",
    "Rural South America",
}

# İngilizce ara etiket → Türkçe nihai etiket
_ETIKET_TR: dict[str, str] = {
    # Doğa ve atmosfer
    "Rivers Streams Water":              "Su Sesleri",
    "Weather":                           "Hava Olayları",
    "Rural Soundscapes":                 "Kırsal Atmosfer",
    "Exterior Atmospheres-Rural Backg":  "Kırsal Atmosfer",
    "Rural Sounds":                      "Kırsal Atmosfer",
    "Exterior Atmosphere":               "Dış Mekan Atmosferi",
    "Interior Atmosphere":               "İç Mekan Atmosferi",

    # Şehir / mekan
    "City Sounds":                       "Şehir Sesleri",
    "Suburbia":                          "Banliyö Atmosferi",
    "Big Ben Taxi Bus Atmospheres":      "Kentsel Atmosfer",

    # Hayvanlar
    "Animals and Birds":                 "Hayvan Sesleri",
    "Birds":                             "Kuş Sesleri",
    "Cats":                              "Kedi Sesleri",
    "Livestock":                         "Çiftlik Hayvanları",
    "Horses":                            "At Sesleri",
    "Horses & Dogs":                     "At ve Köpek Sesleri",

    # Ulaşım
    "Transportation":                    "Ulaşım Sesleri",
    "Auto":                              "Araç Sesleri",
    "Trains":                            "Tren Sesleri",
    "Age Of Steam":                      "Buharlı Tren Sesleri",
    "Aircraft":                          "Uçak Sesleri",
    "Ships And Boats":                   "Gemi ve Tekne Sesleri",

    # İnsan / kalabalık
    "Audiences Children Crowds Foots":   "Kalabalık ve Çocuk Sesleri",
    "Crowds":                            "Kalabalık Sesleri",
    "Schools & Crowds":                  "Okul ve Kalabalık",
    "Footsteps":                         "Ayak Sesleri",
    "Babies":                            "Bebek Sesleri",

    # Ev içi
    "Household":                         "Ev İçi Sesler",

    # Endüstri ve teknoloji
    "Industry":                          "Endüstri Sesleri",
    "Construction":                      "İnşaat Sesleri",
    "Farm Machinery":                    "Tarım Makineleri",
    "Computers Printers Phones":         "Bilgisayar ve Ofis Sesleri",
    "Electronically Generated Sounds":   "Elektronik Sesler",

    # Olay / acil
    "Explosions Guns Alarms":            "Patlama Silah ve Alarm",
    "Emergency":                         "Acil Servis Sesleri",

    # Spor
    "Sport Leisure":                     "Spor ve Eğlence",
    "Adventure Sports":                  "Macera Sporları",
    "Equestrian Events":                 "Binicilik Etkinlikleri",

    # Sağlık
    "Hospitals":                         "Hastane Sesleri",

    # Genel
    "BBC Sound Effects":                 "Genel Efektler",
    "Misc":                              "Çeşitli Sesler",
}


def _bbc_prefix_kaldir(ham: str) -> str:
    return re.sub(r"^BBC\s*\d+\s*[-_]\s*", "", ham).strip()


def _varyantlari_birlestir(etiket: str) -> str:
    # Sondaki Roma rakamı (I, II, III, IV …)
    etiket = re.sub(r"\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\s*$", "", etiket).strip()
    # Sondaki sayı (1, 2 …)
    etiket = re.sub(r"\s+\d+\s*$", "", etiket).strip()

    if etiket in _SEHIR_KENTSEL:
        return "City Sounds"
    if etiket in _KIRSAL_DOGAL:
        return "Rural Sounds"

    return etiket


def turkce_etiket(ham_etiket: str) -> str:
    """
    BBC klasör adından (örn. 'BBC 55 - Footsteps 1') Türkçe nihai etikete
    (örn. 'Ayak Sesleri') dönüştürür.
    """
    ara = _varyantlari_birlestir(_bbc_prefix_kaldir(ham_etiket))
    return _ETIKET_TR.get(ara, ara)
