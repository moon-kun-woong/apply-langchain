from flask import Flask, request ,jsonify,get_flashed_messages
from flask_restx import Resource, Api
import requests
from templates.list_task_command import give_task_list
from templates.add_task_command import add_task_process

app = Flask(__name__)
api = Api(app)

username = {}
count = 1


@api.route("/api/listTask/<string:userId>")
class ListTaskCommand(Resource):
    def get(self, userId):
        response = requests.get(f'http://localhost:8080/api/task/get/{userId}').text
        list_task_command = give_task_list(response)
        list_Values = list_task_command.get("text")
        return jsonify({"text":list_Values})
    
@api.route("/api/addTask", methods=["POST"])
class AddTaskCommand(Resource):
    def post(self):
        data = request.get_json(self)
        add_command_processing = add_task_process(data)
        return jsonify({"response":add_command_processing})

@api.route("/api/task-priority/<string:priority>")
class RequestPriorityValue(Resource):
    def get(self, priority):
        request_priority = requests.get(f'http://localhost:8080/api/task/taskPriority/{priority}')
        print(request_priority)
        return request_priority

if __name__ == '__main__':  
   app.run('0.0.0.0',port=5000,debug=True)
   