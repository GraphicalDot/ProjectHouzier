#!/usr/bin/env python
import requests
import pymongo
connection = pymongo.Connection()
android_db = connection.android_app
android_users = android_db.users
users_pic = android_db.pics

URL = "http://localhost:8000"



class api_tests():
        """
        This function tests all the apis present on this file
        """

        def __init__(self):

                print "\t\t Testing /fb_login api \n\n"
                self.url = "{0}/fb_login".format(URL)
                self.post_user()
                self.test_fb_login_existing_user()
                self.test_fb_login_update_user_friends()
                self.remove_test_user()
                
                print "\t\t Testing /post_comment api \n\n"
                self.url = "{0}/post_comment".format(URL)
                self.test_post_comment_user_doesnt_exist()
                self.insert_test_user()
                self.test_post_comment_another_comment_post()
                self.test_post_comment_another_comment_post()
                self.test_post_comment_check_another_comment_in_db()
                self.remove_test_user()


                return     


        def insert_test_user(self):
                """
                This doesnt test anything but just be used to insert test user
                """
                r = requests.post('http://localhost:8000/fb_login', 
                            data={"fb_id": "442424", 
                                "email": None, 
                                "user_name": "kaali", 
                                "user_friends": "hey", 
                                "location": "delhi", 
                                "date_of_birth": "20-june-1986", 
                                "gender": "male"})
                return 




        def post_user(self):
                TEST = "Post new user"
                r = requests.post('http://localhost:8000/fb_login', 
                            data={"fb_id": "442424", 
                                "email": None, 
                                "user_name": "kaali", 
                                "user_friends": "hey", 
                                "location": "delhi", 
                                "date_of_birth": "20-june-1986", 
                                "gender": "male"})
                
                print ["Test Failed {0}\n".format(TEST), "Test Passed {0}\n".format(TEST)][r.json()["success"] == True]
                return 


        def test_fb_login_existing_user(self):
                TEST = "Check existing user"
                r = requests.post(self.url,
                            data={"fb_id": "442424",
                                "email": "saurav.verma@outlook.com",
                                "user_name": "saurav verma",
                                "user_friends": ["112", "113", "114"]})
                print ["Test Failed {0}\n".format(TEST), "Test Passed {0}\n".format(TEST)][r.json()["success"] == False]
                return 
        
        def test_fb_login_update_user_friends(self):
                TEST = "Check updated user friends list"
                r = requests.post(self.url,
                            data={"fb_id": "442424",
                                "email": "saurav.verma@outlook.com",
                                "user_name": "saurav verma",
                                "user_friends": ["112", "113", "114", "115"]})
        
                print ["Test Failed {0}\n".format(TEST), "Test Passed {0} \n".format(TEST)][r.json()["success"] == True]
                return

        def remove_test_user(self):
                android_users.remove({"fb_id": "442424"})
                return

        def test_post_comment_user_doesnt_exist(self):
                TEST = "Check post comment user doesnt exist"
                r = requests.post(self.url,
                        data={"fb_id": "442424", 
                            "comment": "I dont know why i keep posting the comments again and again"})
        
                print ["Test Failed {0}\n".format(TEST), "Test Passed {0} \n".format(TEST)][r.json()["success"] == False]
                return
        
        def test_post_comment_comment_post(self):
                TEST = "Check post comment first comment posted"
                r = requests.post(self.url,
                        data={"fb_id": "442424", 
                            "comment": "I dont know why i keep posting the comments again and again"})
        
                print r.json()["messege"]
                print ["Test Failed {0}\n".format(TEST), "Test Passed {0} \n".format(TEST)][r.json()["success"] == True]
                return
        
        def test_post_comment_another_comment_post(self):
                TEST = "Check post comment second comment posted"
                r = requests.post(self.url,
                        data={"fb_id": "442424", 
                            "comment": "Another comment posted"})
        
                print ["Test Failed {0}\n".format(TEST), "Test Passed {0} \n".format(TEST)][r.json()["success"] == True]
                return

        def test_post_comment_check_another_comment_in_db(self):
                TEST = "Check post comment posted in mongodb"
                if android_users.find_one({"comments": {"$in": ["Another comment posted"]}}, {"fb_id": "442424"}):
                        print "Test passed, Comment exists in mongodb"




if __name__ == "__main__":
        __test = api_tests()

