import re

def formatter(sel):

    res = {}

    _timeval = sel.xpath('//*[@id="root"]/div/div/section/div/div/header/p/text()').extract_first()
    timeval = re.findall(r'\((.+)\u5206|\((.+)\u5206\u4EE5\u4E0A', _timeval)
    if len(timeval)!=0:
        timeval = int(timeval[0][0].translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})))
    else:
        timeval = re.findall(r'\((.+)\u6642\u9593|\((.+)\u6642\u9593\u4EE5\u4E0A', _timeval)
        timeval = int(timeval[0][0].translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}))) * 60
    res["time"] = timeval

    #timeval = re.findall(r'\((.+)\u5206|\((.+)\u5206\u4EE5\u4E0A|\((.+)\u6642\u9593|\((.+)\u6642\u9593\u4EE5\u4E0A', timeval)[0][0]
    #n_components = int(len(sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[1]/tbody/tr/td/text()').extract()) / 2)

    n_howtomake = int(len(sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[2]/tbody/tr/td/text()').extract()) / 2)
    res["n_howtomake"] = n_howtomake

    k_components = sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[1]/tbody/tr/td[1]/text()').extract()
    v_components = sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[1]/tbody/tr/td[2]/text()').extract()
    components = {}
    for l, val in enumerate(k_components):
        components[val.strip()] = v_components[l].strip()
    res["n_components"] = len(k_components)
    res["components"] = components


    k_howtomake = sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[2]/tbody/tr/td[1]/text()').extract()
    v_howtomake = sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[1]/table[2]/tbody/tr/td[2]/text()').extract()
    howtomake = {}
    for l, val in enumerate(k_howtomake):
        howtomake[val.strip()] = v_howtomake[l].strip()
    res["n_howtomake"] = len(k_howtomake)
    res["howtomake"] = howtomake

    res["title"] = sel.xpath('//*[@id="root"]/div/div/section/div/div/header/p/text()').extract_first().strip().replace('\n','').replace(' ','')
    res["description"] = sel.xpath('//*[@id="root"]/div/div/section/div/div/div[1]/figure/figcaption/p/text()').extract_first().strip()

    point = sel.xpath('//*[@id="root"]/div/div/section/div/div/div[2]/div[2]/text()').extract()
    point = "".join(point).strip()
    res["point"] = point

    return res
