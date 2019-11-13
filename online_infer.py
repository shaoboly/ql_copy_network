from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import logging
import os
from math import inf

import numpy as np
import tensorflow as tf

import config
from data_reading import processors
from data_reading.batcher import Batcher, EvalData
from inference import inference
from model_pools import modeling, model_pools
from utils.metric_utils import calculate_reward,convert_pred_to_token,calc_acc
from utils.utils import print_parameters, pprint_params

from inference import *

from online_service.online_model import online_model

import web
from web import form

import numpy as np
import os


render = web.template.render('templates/')

urls = (
  '/', 'index',
  '/tg','json_service'
)

log = open("log.txt","a",encoding="utf-8")

topic_form = form.Form(
    # form.Textbox("email", vemail, description="E-Mail"),
    # form.Password("password", vpass, description="Password"),
    # form.Password("password2", description="Repeat password"),
    form.Textarea("query", description="input_text"),
    form.Textarea("parent", description="parent",
                  readonly="readonly"),
    form.Textarea("child", description="child",
                  readonly="readonly"),
    form.Button("submit", type="submit", description="Register"),
    # validators = [
    #    form.Validator("Passwords did't match", lambda i: i.password == i.password2)]

)




class index:
    def GET(self):
        f = topic_form()
        f['query'].value = "tom hanks acted movie"
        #f['parent'].value = "film film music film film music"
        return render.topic(f)

    def POST(self):
        post_value = web.input(query=None,parent=None)
        f = topic_form()

        f['query'].value = str(post_value.query)
        #f['parent'].value = str(post_value.parent)

        print(str(post_value.query))
        #print(str(post_value.parent))
        if post_value.query == None:
            return render.topic(f)

        query = str(post_value.query)
        #lf = str(post_value.parent)
        result = online_model.infer_one_sentence(query)

        result= str(result)

        #if len(results)==0:
        #    final_result="No Trigger"
        #else:
        #    final_result = "\n".join(results)

        log.write(query+'\n')
        log.write(result+'\n')
        log.write("========================================\n\n")
        log.flush()

        f['parent'].value = result
        return render.topic(f)


import json
class json_service:
    def POST(self):
        post_value = web.input(query=None)
        json_result = {}
        if post_value.query == None:
            json_result["status"] = "NoQuery"
            json_reuslt = json.dumps(json_result)
            return json_reuslt
        print("query:" + str(post_value.query))
        query = str(post_value.query)

        query = query.strip()
        # lf = str(post_value.parent)
        result = online_model.infer_all_candidates(query)

        result = str(result)

        json_result["query"] = post_value.query
        json_result["result"] = str(result)
        json_result["status"] = "Success"
        json_reuslt = json.dumps(json_result)

        print(json_result)
        return json_reuslt

    def GET(self):
        post_value = web.input(query=None)
        json_result = {}
        if post_value.query == None:
            json_result["status"] = "NoQuery"
            json_reuslt = json.dumps(json_result)
            return json_reuslt
        print("query:" + str(post_value.query))
        query = str(post_value.query)

        query = query.strip()
        # lf = str(post_value.parent)
        result = online_model.infer_all_candidates(query)

        result = str(result)

        json_result["query"] = post_value.query
        json_result["result"] = str(result)
        json_result["status"] = "Success"
        json_reuslt = json.dumps(json_result)

        print(json_result)
        return json_reuslt

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()