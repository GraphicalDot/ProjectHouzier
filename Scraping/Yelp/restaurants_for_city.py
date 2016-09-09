#!/usr/bin/env python
import requests
import BeautifulSoup
import warnings
import sys
import os
import hashlib
import time
from selenium import webdriver
import pymongo
import re
from compiler.ast import flatten


file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(file_path)
from GlobalConfigs import bcolors, MONGO_REVIEWS_IP, MONGO_REVIEWS_IP, MONGO_REVIEWS_PORT,\
        MONGO_YELP_DB, MONGO_YELP_EATERIES, MONGO_YELP_REVIEWS

                                                                                                                                                       
def encoding_help(obj):
        if not isinstance(obj, unicode):
                obj = unicode(obj)
        obj = obj.encode("ascii", "xmlcharrefreplace")
        return obj 






class PerCity:
        def __init__(self, url, start, end, do_print):
                self.url = url 

                r = requests.get(self.url)
                self.url_soup = BeautifulSoup.BeautifulSoup(r.text)
                self.start = start
                self.end = end
                self.__print = (True, False)[do_print == False]

        def __number_of_pages(self):
                number_of_pages = self.url_soup.find("div", {"class": "page-of-pages arrange_unit arrange_unit--fill"}).text.replace('Page 1 of ', "")
                if self.__print:
                        print number_of_pages
                return number_of_pages

    
        def __make_links(self):
                links = list()
                pages = self.__number_of_pages()
                links.append(self.url)

                for number in  range(1, int(pages) + 1):
                        links.append("{0}&start={1}0".format(self.url, number))
                if self.__print:
                        print links
                return links


        def __filter_links(self):
                links_to_be_scraped = self.__make_links()[int(self.start): int(self.end)]
                if self.__print:
                        print links_to_be_scraped
                return links_to_be_scraped



        def scrape_links(self):
                restaurants = list()
                links_to_be_scraped = self.__filter_links()
                for link in links_to_be_scraped:
                        restaurants.extend(self.__per_page_data(link))

                if self.__print:
                        print restaurants

                self.restaurants = restaurants
                return restaurants

        def __open_and_soup(self, link):
                driver = webdriver.Firefox()
                driver.get(link)
                html = driver.page_source
                driver.close()
                return BeautifulSoup.BeautifulSoup(html)
        
        def __per_page_data(self, page_url):
                """
                Extracts resturant information from one page which approximately yelp 
                has 10 restaurants per page
                """
                __per_page_restaurants = list()

                """
                __html = requests.get(page_url)
                
                __soup = BeautifulSoup.BeautifulSoup(__html.text)
                """
                __soup = self.__open_and_soup(page_url)

                for __res in __soup.findAll("div", {"class": "biz-listing-large"}):
                        try:
                                res_dict = self.__each_restaurant_data(__res)
                                
                                ##TO remove any page that has been classified as ad
                                if not re.search("ad_business", res_dict.get("eatery_link")):
                                        __per_page_restaurants.append(res_dict)
                        except StandardError as e:
                                print e
                                pass


                return __per_page_restaurants

        def catch_exception(func):
                def deco(self, review):
                        try:
                                return func(self, review)
                        except Exception as e:
                                warnings.warn("{color} ERROR <{error}> in function <{function}> {Reset}".format(\
                                        color=bcolors.FAIL, error=e, function=func.__name__ , Reset=bcolors.RESET))
                                  
                                return None
                return deco       


        def __each_restaurant_data(self, restaurant_soup):
                res_data = dict()
                res_data["site"] = "yelp"
                res_data["eatery_name"] = self.__res_name(restaurant_soup)
                res_data["eatery_rating"] = self.__res_rating(restaurant_soup)
                res_data["eatery_categories"] = self.__res_categories(restaurant_soup)
                res_data["eatery_phone_number"]= self.__res_phone_number(restaurant_soup)
                res_data["eatery_city"] = self.__res_city(restaurant_soup)
                res_data["eatery_address"] = self.__res_address(restaurant_soup)
                res_data["eatery_link"] = self.__res_link(restaurant_soup)
                res_data["eatery_total_reviews"] = self.__res_reviews(restaurant_soup)
                res_data["cost_scale"] = self.__cost_scale(restaurant_soup)
                
                try:
                        __str = encoding_help(res_data["eatery_name"]) + res_data["eatery_city"] + res_data["eatery_address"]
                        res_data["eatery_id"] = hashlib.md5(__str).hexdigest()
                        return res_data
                except Exception as e:
                        raise StandardError("{color} Eatery with link {link} and name {name} couldnt be scraped {Reset}".format(
                                    color=bcolors.FAIL, link = res_data["eatery_link"], name=res_data["eatery_name"], 
                                    Reset = bcolors.RESET))
               
        @catch_exception
        def __cost_scale(self, __soup):
                return len(__soup.find("span", {"class": "business-attribute price-range"}).text)
        
        @catch_exception
        def __res_name(self, __soup):
                return __soup.find("a", {"class": "biz-name"}).text
                
        @catch_exception
        def __res_rating(self, __soup):
                return __soup.find("div", {"class": "rating-large"}).find("i")["title"].replace(" star rating", "")

        @catch_exception
        def __res_categories(self, __soup):
                return [__a.text for __a in __soup.find("span", {"class": "category-str-list"}).findAll("a")]
                
        @catch_exception
        def __res_phone_number(self, __soup):
                return __soup.find("span", {"class": "biz-phone"}).text

        @catch_exception
        def __res_city(self, __soup):
                return __soup.find("div", {"class": "secondary-attributes"}).\
                        find("span", {"class": "neighborhood-str-list"}).text
                
        @catch_exception
        def __res_address(self, __soup):
                return __soup.find("div", {"class": "secondary-attributes"}).find("address").text
        
        @catch_exception
        def __res_reviews(self, __soup):
                return __soup.find("span", {"class": "review-count rating-qualifier"}).text.replace(" reviews", "")
    
        @catch_exception
        def __res_link(self, __soup):
                return "http://www.yelp.com{0}".format(__soup.find("a", {"class": "biz-name"}).get("href"))


        def insert_in_db(self):
                __db_instance = YelpMongoDB()
                for eatery_dict in self.restaurants:
                        print eatery_dict
                        __db_instance.push_eatery(eatery_dict)
                __db_instance.close_connection()
                return self.restaurants
                        


class PerRestaurant:
        def __init__(self, eatery_dict):
                """
                Args:
                    res_dict:
                        {'eatery_city': u'Nolita', 'eatery_phone_number': u'(212) 226-6378', 
                        'eatery_rating': u'4.5', 'site': 'yelp', 'eatery_total_reviews': u'232', 
                        'eatery_categories': [u'Bakeries', u'Sandwiches'], 
                        'eatery_address': u'198 Mott StNew York, NY 10012', 
                        'eatery_link': 'http://www.yelp.com/biz/parisi-bakery-new-york', 
                        'eatery_id': 'cae8eac62271c948e3e4453c585a0883', 'eatery_name': u'Parisi Bakery', 
                        'cost_scale': 1}
                        
                """
                self.eatery_dict = eatery_dict
                self.main_page_soup = self.__open(self.eatery_dict.get("eatery_link"))
                self.no_of_review_pages = self.main_page_soup.find("div",  {"class": "page-of-pages arrange_unit arrange_unit--fill"}).\
                        text.replace("Page 1 of ", "")
                self.eatery_id = self.eatery_dict.get("eatery_id")
                self.date_pattern = "%Y-%m-%d"


        def __additional_eatery_detail(self):
                more_details = dict()
                for e in self.main_page_soup.find("div", {"class": "short-def-list"}).findAll("dl"):
                            more_details.update({
                                e.findNext().text : e.findNext().findNext().text
                                })
                return more_details

        
        def __menu(self):
                def cuisine_name(__soup):
                        return __soup.find("h3").text

                def cuisine_description(__soup):
                        try:
                                return __soup.find("p").text
                        except Exception:
                                return None

                def cuisine_amount(__soup):
                        try:
                                return __soup.find("li", {"class": "menu-item-price-amount"}).text
                        except Exception:
                                return None
                                            
                
            
                self.menu_list = list()
                self.menu_link = self.eatery_dict.get("eatery_link").replace("biz", "menu")
                soup = self.__open(self.menu_link)

                if soup.find("body", {"class": "ytype error-page"}):
                        warnings.warn("{0}This eatery doesnt have any menu uploaded yet {1}".format(bcolors.FAIL, bcolors.RESET))

                try:
                        for menu_section in soup.findAll("div", {"class": "menu-section"}):
                                for __soup in menu_section.findAll("div", {"class": "media-story"}):
                                            self.menu_list.append({"name": cuisine_name(__soup),
                                                    "description": cuisine_description(__soup),
                                                    "cost": cuisine_amount(__soup),
                                        })
               

                except AttributeError as e:
                        warnings.warn("{0}THe required html tags couldt be found {1}".format(bcolors.FAIL,  bcolors.RESET))


                print self.menu_list
                return self.menu_list                        


        def catch_exception(func):
                def deco(self, review):
                        try:
                                return func(self, review)
                        except Exception as e:
                                warnings.warn("{color} ERROR <{error}> in function <{function}> {Reset}".format(\
                                        color=bcolors.FAIL, error=e, function=func.__name__ , Reset=bcolors.RESET))
                                  
                                return None
                return deco       
    
        def __open(self, link):
                driver = webdriver.Firefox()
                driver.get(link)
                html = driver.page_source
                driver.close()
                return BeautifulSoup.BeautifulSoup(html)


        def __make_links(self):
                """
                Excludes the first link which was gieven to this class already
                """
                pages = list()
                i = 40
                ##Appending first page to the list of pages from which the reviews need to be scraped
                pages.append("{0}?sort_by=date_desc".format(self.eatery_dict.get("eatery_link")))
                for page in range(1, int(self.no_of_review_pages)):
                        pages.append("{0}?start={1}&sort_by=date_desc".format(self.eatery_dict["eatery_link"], i))
                        i+= 40
                return pages

        def __scrape_reviews(self):
                __db_instance = YelpMongoDB()

                """
                if not __db_instance.check_for_last_review(self.eatery_id):
                        print "{0} All the reviews has aleady been scraped {1}".format(bcolors.OKGREEN, bcolors.RESET)
                        __db_instance.close_connection()
                        return list()
                """
                #From the original link, scrape reviews
                all_reviews = list()
                for page_link in self.__make_links():
                        print "Now scraping %s"%page_link
                        __soup = self.__open(page_link)
                        __reviews = self.__per_page_reviews(__soup)
                        all_reviews.extend(__reviews)
                return all_reviews
                


        def run(self, insert_in_mongodb=False):
                """
                Updates the eatery dict with additional information 
                Updates the eatery dict with menu
                Scrape all reviews for eatery dict
                insert in mongodb if insert_in_mongodb
                """
                self.eatery_dict.update({"additional_details": self.__additional_eatery_detail()})
                self.eatery_dict.update({"menu": self.__menu()})
                
                print self.eatery_dict
                reviews = self.__scrape_reviews()
                
                if insert_in_mongodb:
                        __db_instance = YelpMongoDB()
                        __db_instance.push_eatery(self.eatery_dict)
                        for review_dict in reviews:
                                __db_instance.push_review(review_dict)
                        __db_instance.close_connection()

                return


        def __per_page_reviews(self, page_soup):
                review_list = list()
                reviews = page_soup.find("ul", {"class": "ylist ylist-bordered reviews"}).findAll("div", {"class": "review-wrapper"})
                for review_soup in reviews:
                        review_dict = dict()
                        date_published = self.__date_published(review_soup)
                        converted_epoch = int(time.mktime(time.strptime(date_published, self.date_pattern)))


                        review_dict.update({"readable_review_day": self.__readable_review_day(date_published)}) 
                        review_dict.update({"readable_review_month": self.__readable_review_month(date_published)}) 
                        review_dict.update({"readable_review_year": self.__readable_review_year(date_published)}) 
                        review_dict.update({"converted_epoch": converted_epoch}) 
                        review_dict.update({"review_time": date_published}) #u'2014-10-20'



                        review_text = encoding_help(self.__review_text(review_soup))

                        review_dict.update({"review_rating": self.__rating_value(review_soup)})
                        review_dict.update({"review_text": review_text})
                        review_dict.update({"scraped_epoch": self.__scraped_epoch()})
                        
                        review_dict.update({"review_id": hashlib.md5(review_text + self.eatery_id).hexdigest()})
                        
                        review_dict.update({"eatery_id": self.eatery_id})

                        review_dict.update(self.__votings(review_soup))
                        review_list.append(review_dict)
                return review_list


        @catch_exception
        def __date_published(self, review_soup):
                return review_soup.find("meta", {"itemprop": "datePublished"})["content"]

        @catch_exception
        def __rating_value(self, review_soup):
                return review_soup.find("meta", {"itemprop": "ratingValue"})["content"]

        @catch_exception
        def __review_text(self, review_soup):
                __text = review_soup.find("p", {"itemprop": "description"}).text
                if not __text:
                        raise StandardError("Review text couldnt be found")
                return __text

        def __scraped_epoch(self):
                return int(time.time())

        def __readable_review_day(self, date_published):
                return date_published.split("-")[2]

        def __readable_review_month(self, date_published):
                return date_published.split("-")[1]
        
        def __readable_review_year(self, date_published):
                return date_published.split("-")[0]


        @catch_exception
        def __votings(self, review_soup):
                voting_dict = dict()
                for button in review_soup.findAll("li", {"class": "vote-item inline-block"}):
                        voting_dict.update({
                            button.find("span", {"class": "vote-type"}).text: button.find("span",  {"class": "count"}).text})

                return voting_dict


class YelpMongoDB:
        
        def __init__(self):
                self.connection = pymongo.MongoClient(MONGO_REVIEWS_IP, MONGO_REVIEWS_PORT)
                self.eateries = eval("self.connection.{db_name}.{collection_name}".format(db_name=MONGO_YELP_DB,
                    collection_name=MONGO_YELP_EATERIES))
               
                print self.eateries
                self.reviews = eval("self.connection.{db_name}.{collection_name}".format(db_name=MONGO_YELP_DB,
                    collection_name=MONGO_YELP_REVIEWS))



        def push_eatery(self, eatery_dict):
                """
                {'eatery_city': u'West Village', 'eatery_phone_number': u'(212) 462-0041', 'eatery_rating': u'4.0', 
                'site': 'yelp', 'eatery_id': '38d7b66c0a62046c1b19a2accfd4f6d6', 'eatery_categories': u'4.0', 
                'eatery_address': u'49 Carmine StNew York, NY 10014', 
                'eatery_link': 'http://www.yelp.com/biz/the-grey-dog-new-york-6', 'eatery_name': u'The Grey Dog'},
                'eatery_total_reviews': u'193'
                """
                eatery_id = eatery_dict.get("eatery_id")
                if self.eateries.find_one({"eatery_id": eatery_id}):
                        self.eateries.update({"eatery_id": eatery_id}, {"$set": {"eatery_rating": eatery_dict.get("eatery_rating"),
                                        "eatery_categories": eatery_dict.get("eatery_categories"),
                                        "eatery_total_reviews": eatery_dict.get("eatery_total_reviews"),
                                        "menu": eatery_dict.get("menu"), 
                                        "additional_details": eatery_dict.get("additional_details"),
                                }}, upsert=False, multi=False)
                        return 

                self.eateries.insert(eatery_dict)
                return 


        def close_connection(self):
                self.connection.close()
                return


        def check_for_last_review(self, eatery_id):
                if not bool(list(self.eateries.find({"review_id": eatery_id}))):
                        return False


                last_epoch = yelp_reviews.find({"eatery_id": 1}, fields={"_id": 0, "converted_epoch": 1})\
                                                    .sort("converted_epoch",  -1).limit(1)
        
                return last_epoch



        def push_review(self, review_dict):
                if not self.reviews.find_one({"review_id": review_dict.get("review_id") }):
                        self.reviews.insert(review_dict)    
                        return 
                print "{0}Review has already been found in the database {1}".format(bcolors.FAIL, bcolors.RESET)
                return 


if __name__ == "__main__":
        instance = PerCity("http://www.yelp.com/search?find_loc=New+York%2C+NY&cflt=food", 0, 1, False)
        eateries = instance.scrape_links()
        """
        instance.insert_in_db()
        eatery_dict = {'eatery_city': u'Nolita', 'eatery_phone_number': u'(212) 226-6378', 
                        'eatery_rating': u'4.5', 'site': 'yelp', 'eatery_total_reviews': u'232', 
                        'eatery_categories': [u'Bakeries', u'Sandwiches'], 
                        'eatery_address': u'198 Mott StNew York, NY 10012', 
                        'eatery_link': 'http://www.yelp.com/biz/parisi-bakery-new-york', 
                        'eatery_id': 'cae8eac62271c948e3e4453c585a0883', 'eatery_name': u'Parisi Bakery', 
                        'cost_scale': 1}

        """
        for eatery_dict in eateries:
                __ins = PerRestaurant(eatery_dict)
                __ins.run(insert_in_mongodb=True)


