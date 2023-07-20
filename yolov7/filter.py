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
    