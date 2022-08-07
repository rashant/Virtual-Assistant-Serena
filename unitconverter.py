import requests_html
def converter(unit):
    session=requests_html.HTMLSession()
    r = session.get("https://www.google.com/search?q=" + unit)
    try:
        article = r.html.find('input.a61j6.vk_gy.vk_sh.Hg3mWc', first=True)
        x=article.attrs['value']
        return x
    except:
        article = r.html.find('#NotFQb', first=True)
        values = article.find('input.vXQmIe.gsrt', first=True)
        z = values.attrs['value']
        z = str(z)
        return z

