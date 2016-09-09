: for sent in place_sentences:
    result = loads(server.parse(sent))
    for element in result["sentences"][0]["words"]:
        if element[1].get('NamedEntityTag') == "LOCATION":
            print element
            print sent
