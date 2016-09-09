#!/usr/bin/env python

{
  "food" : {
    "mappings" : {
      "menu-food" : {
        "properties" : {
          "eatery_id" : {
            "type" : "string"
          },
          "eatery_name" : {
            "type" : "string"
          },
          "negative" : {
            "type" : "long"
          },
          "neutral" : {
            "type" : "long"
          },
          "positive" : {
            "type" : "long"
          },
          "super-negative" : {
            "type" : "long"
          },
          "super-positive" : {
            "type" : "long"
          },
          "timeline" : {
            "type" : "string"
          },
          "total_sentiments" : {
            "type" : "long"
          }
        }
      }
    }
  }
}




food_settings = {
                                "settings": {
                                        "analysis": {
                                                "analyzer": {
                                                        "phonetic_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "whitespace",
                                                            "filter": ["lowercase", "asciifolding", "standard", "custom_metaphone"],
                                                                    },
                                                        "keyword_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "keyword",
                                                            "filter": ["lowercase", "asciifolding"],
                                                                    },
                                                        "shingle_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "ngram_tokenizer",
                                                            "filter": ["lowercase", "asciifolding", "shingle_tokenizer"],
                                                                    },
                                                        "custom_analyzer": {
                                                            "type": "custom",
                                                            "tokenizer" : "ngram_tokenizer",
                                                            "filter": ["lowercase", "asciifolding"],
                                                                    },
                                                        "custom_analyzer_two": {
                                                            "type": "custom",
                                                            "tokenizer" : "limited_tokenizer",
                                                            "filter": ["lowercase", "asciifolding"],
                                                                    },
                                                        "standard_analyzer": {
                                                                "type": "custom", 
                                                                "tokenizer": "standard",
                                                                "filter": ["lowercase", "asciifolding"],
                                                                }
                                                        },
                                                "tokenizer": {
                                                        "ngram_tokenizer": {
                                                                "type" : "edgeNGram",
                                                                "min_gram" : 2,
                                                                "max_gram" : 100,
                                                                "token_chars": [ "letter", "digit" ]
                                                                },
                                                        "limited_tokenizer": {
                                                                "type" : "edgeNGram",
                                                                "min_gram" : "2",
                                                                "max_gram" : "10",
                                                                "token_chars": [ "letter", "digit" ]
                                                                },
                                                        }, 
                                                "filter": {
                                                            "shingle_tokenizer": {
                                                                "type" : "shingle",
                                                                "min_shingle_size" : 2,
                                                                "max_shingle_size" : 5,
                                                                },

                                                            "custom_metaphone": {
                                                                    "type" : "phonetic",
                                                                    "encoder" : "metaphone",
                                                                    "replace" : False
                                                                    }
                                                            }
                                                }
                                        }}
dish_mappings = {'dish': {
                                 '_all' : {'enabled' : True},
                                'properties': 
                                                {'name': 
                                                        {
                                                            #'analyzer': 'custom_analyzer', 
                                                            'type': 'string', 
                                                            'copy_to': ['dish_raw', 'dish_shingle', "dish_phonetic"],
                                                            },
                                                    
                                                    
                                        'dish_phonetic': {
                                                    'type': 'string', 
                                                    'analyzer': 'phonetic_analyzer',
                                                    },
                                        'dish_shingle': {
                                                    'type': 'string', 
                                                    'analyzer': 'shingle_analyzer',
                                                    },
                                        'dish_raw': {
                                                    'type': 'string', 
                                                    'analyzer': 'keyword_analyzer',
                                                    },

                                        'eatery_shingle': {
                                                    'type': 'string', 
                                                    'analyzer': 'shingle_analyzer',
                                                    },

                                        'eatery_raw': {
                                                    'type': 'string', 
                                                    'analyzer': 'keyword_analyzer',
                                                    },
                                        
                                        'negative': {'type': 'long'},
                                        'neutral': {'type': 'long'},
                                        'positive': {'type': 'long'},
                                        'similar': {
                                                'properties': {'name': 
                                                                    {
                                                                        'type': 'string', 
                                                                        'copy_to': ['dish_raw', 'dish_shingle', "dish_phonetic"],
                                                                    
                                                                    },
                                                            'negative': {
                                                                     'type': 'long'},
                                                            'neutral': {
                                                                    'type': 'long'},
                                                            'positive': {
                                                                    'type': 'long'},
                                                            'super-negative': {
                                                                    'type': 'long'},
                                                            'super-positive': {
                                                                    'type': 'long'},
                                                            'timeline': {
                                                                'type': 'string'}
                                                            }
                                                },
   
                                        'super-negative': {
                                                    'type': 'long'},
                                        'super-positive': {
                                                    'type': 'long'},
                                        'eatery_name': {
                                                            'type': 'string', 
                                                            'copy_to': ['eatery_shingle', "eatery_raw"],
                                                    },
                                        'eatery_id': {
                                                    'type': 'string', 
                                                    'index': 'not_analyzed',
                                                    },
                                        'total_sentiment': {
                                                    'type': 'integer', 
                                                    },
                                        'timeline': {
                                            'type': 'string'}}}}
                

