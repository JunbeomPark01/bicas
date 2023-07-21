def confidence(li, pass_conf, amb_conf):
    filter = "delete"
    #print(li)
    passList = [x for x in li if x > pass_conf]
    #print(passList)
    ambList = [x for x in li if amb_conf <= x and x < pass_conf]
    #print(ambList)
    failList = [x for x in li if x < amb_conf]
    #print(failList)
    returnList = []
    if len(passList)>0:
        filter = "pass"
        returnList = passList
    if len(ambList)>0:
        filter = "amb"
        returnList = ambList
    if len(failList)>0:
        filter = "fail"
        returnList = failList
    return filter, returnList
    
    #conf_sum = sum(li)/len(li)
    
def sum_confidence(conf_sum, pass_conf, amb_conf):
    result = "delete"
    returnList = []

    if isinstance(conf_sum, (int, float)):  # 단일 값인 경우 리스트로 변환
        conf_sum = [conf_sum]

    pass_list = list(filter(lambda x: x > pass_conf, conf_sum))
    amb_list = list(filter(lambda x: amb_conf <= x < pass_conf, conf_sum))
    fail_list = list(filter(lambda x: x < amb_conf, conf_sum))

    if len(pass_list) > 0:
        result = "pass"
        returnList = pass_list
    if len(amb_list) > 0:
        result = "amb"
        returnList = amb_list
    if len(fail_list) > 0:
        result = "fail"
        returnList = fail_list

    return result, returnList

