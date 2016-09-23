
#!/usr/bin/env python
import jsonrpc
from simplejson import loads
import pprint 
import blessings

terminal = blessings.Terminal()

class CoreNLPScripts(object):
        def __init__(self):
                pass


        @staticmethod
        def store_sentiment_sentences_analysis(self,
                                               key_name= None, 
                                               source_collection_name=None,
                                               target_collection_name=None,
                                               corenlp_server_connection=None,
                                               refresh=False):
                    """
                    This class method Interacts with cornlp server and save the
               
                    

                    """
                    assert(key_name), "The key_name is the key under which\
                        the text is stored in each document for source_collection_name"
                    assert(target_collection_name), "target collection name should be\
                        provided to store the results"
                    assert(source_collection_name), "collection name should be\
                        provided"
                    assert(corenlp_server_connection), "collection name should be\
                        provided and must be running"

                    
                    for sentence in source_collection_name.find():
                            CoreNLPScripts.sentence_analysis(sentence.get(key_name), 
                                                            corenlp_server_connection)
                            try:
                                    target_collection_name.insert({"result": result,
                                                               "setence": sentence})
                            except Exception as e:
                                    terminal.red(e)
                                    pass

                    return 
                    


        @staticmethod
        def sentence_analysis(self, text, corenlp_server_connection):
                    try:
                            result = loads(corenlp_server_connection.parse(text))
                    except Exception as e:
                            print terminal.red(e)
                    
                    print "text %s"%result
                    print terminal.green(pprint.pprint(result))
                    return result



