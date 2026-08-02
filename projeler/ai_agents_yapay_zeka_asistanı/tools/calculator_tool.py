"""
temel bir matematik hesaplayıcı
"""

from langchain.tools import tool #decorator

@tool("calculator")
def calculator(expression: str) -> str: #expression'u str olarak alır ve sonucu string olarak return eder
    """
    Basit matematiksel ifadeleri değerlendir.
    örnek: '25 * (5 + 3)' -> 'Answer: 200'
    """
    try:
        result= eval(expression)
        # "Answer" kelimesi react ajanları için kritik. sonucu doğru anlamasını sağlar
        return f"Answer: {result}" #result string olarak return edilir çünkü llm'in anlaması lazım
    
    except Exception as e:
        return f"Error: {e}"

#print(calculator.invoke('25 * (5 + 3)'))

#langchain formatına uygun hale getir
calculator_tool= calculator

