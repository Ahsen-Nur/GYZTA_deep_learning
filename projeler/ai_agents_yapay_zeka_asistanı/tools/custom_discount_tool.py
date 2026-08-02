"""
ürün fiyatını alır ve %10 indirim uygular
"""

from langchain.tools import tool #decorator
import re #metin içerisinden sayısal fiyat bilgisi çekmek için regex kullanılır

@tool("discount_calculator")
def discount_calculator(product_info: str) -> str:
    """
    ürün fiyatını alır ve %10 indirim uygular
    örnek: "elma fiyatı 50 TL" -> "indirimli fiyat: 45 TL"
    """

    try:
        price= float(re.findall(r"\d+", product_info)[0])
        discounted= price * 0.9
        return f"indirim uygulandı! yeni fiyat: {discounted: .2f} TL"
    except Exception as e:
        return f"Hata: Fiyat bulunamadı. ({e})"


#print(discount_calculator.invoke("elma fiyatı 50 TL"))

discount_tool= discount_calculator
