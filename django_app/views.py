from django.shortcuts import render
from django.http import HttpResponse
from django_app.models import Movies, Ratings
from django.core import serializers
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import torch
import torch.nn as nn
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import base64
import io
from io import BytesIO
from PIL import Image

# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@csrf_exempt
def showDataRating(request):
    if request.method == 'GET':
        # request_body = json.loads(request.body.decode('utf-8'))
        # userid = request_body['user_id']
        userid = request.GET.get('user_id')
        if userid is not None and userid != '':
            result_rating = Ratings.objects.filter(user_id=userid)
        else:
            result_rating = Ratings.objects.all()
        return render(request, "django_app/page2.html", {'Ratings': result_rating})

@csrf_exempt
def showDataMovie(request):
    movieid = request.GET.get('movie_id')
    if movieid is not None and movieid != '':
        result_movie = Movies.objects.filter(movie_id=movieid)
    else:
        result_movie = Movies.objects.all()
    return render(request, "django_app/page1.html", {'Movies': result_movie})

@csrf_exempt
def showRatingJson(request):
    if request.method == "POST":
        request_body = json.loads(request.body.decode('utf-8'))
        userid = request_body['user_id']
        print(userid)
        final_list = []
        result_rating = Ratings.objects.filter(user_id=userid)
        print(result_rating)

        for row in range(0,len(result_rating)):
            row= int(row)
            data = {
                'movie_id': result_rating[row].movie_id,
                'rating': result_rating[row].rating
            }
            final_list.append(data)
        return JsonResponse({"final_list": final_list})

@csrf_exempt
def showMovieJson(request):
    if request.method == "POST":
        request_body = json.loads(request.body.decode('utf-8'))
        movieid = request_body['movie_id']
        movie_list = []
        result_movie = Movies.objects.filter(movie_id=movieid)
        print(result_movie)
        for row in range(0,len(result_movie)):
            row= int(row)
            data = {
                'title': result_movie[row].title,
                'genres': result_movie[row].genres
            }
            movie_list.append(data)
        return  JsonResponse({'movie_list': movie_list})

@csrf_exempt
def insertRating(request):
    if request.method == 'POST':
        request_body = json.loads(request.body.decode('utf-8'))
        userid = request_body['user_id']
        movieid = request_body['movie_id']
        rating = request_body['rating']
        new_rating = Ratings(user_id=userid, movie_id=movieid, rating=rating)
        new_rating.save()
        return JsonResponse({'status': 'success'})

@csrf_exempt
def deleteRating(request):
    if request.method == 'POST':
        request_body = json.loads(request.body.decode('utf-8'))
        id_row = request_body['id']
        instance = Ratings.objects.filter(id=id_row)
        instance.delete()
        return JsonResponse({'status': 'success'})

@csrf_exempt
def updateRating(request):
    if request.method == 'POST':
        request_body = json.loads(request.body.decode('utf-8'))
        id_row = request_body['id']
        new_value = request_body['rating']
        print(id_row, new_value)
        update_row = Ratings.objects.get(id=id_row)
        update_row.rating = new_value
        update_row.save()
        return JsonResponse({'status': 'success'})


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.trans = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        print(x.shape)
        f_x = self.conv1(x)
        f_x = self.bn1(f_x)
        f_x = self.relu(f_x)
        f_x = self.conv2(f_x)
        f_x = self.bn2(f_x)
        print(f_x.shape)

        x = self.trans(x)

        x = f_x + x
        x = self.relu(x)
        return x

def resnet_block(in_channels, out_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(ResidualBlock(in_channels, out_channels))
        else:
            blk.append(ResidualBlock(out_channels, out_channels))
    return blk


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveMaxPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))


# def stringToRGB(base64_string):
#     imgdata = base64.b64decode(str(base64_string))
#     image = Image.open(io.BytesIO(imgdata))
#     return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

@csrf_exempt
def predictImage(request):
    PATH = '/Users/dangquang251197/cifar_net.pth'
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(request.body)
    net.load_state_dict(torch.load(PATH))
    request_body = json.loads(request.body.decode('utf-8'),strict=False)
    # image = request.POST.get('image')
    image = request_body['image']
    print(image)
    image1 = base64.b64decode(image)
    image2 = np.fromstring(image1, dtype= np.uint8)
    image3 = cv2.imdecode(image2, cv2.IMREAD_COLOR)
    image4 = torch.tensor(image3)
    image4 = image4.permute(2, 0, 1)
    image4 = image4.unsqueeze(0)
    image4 = image4.float()
    outputs = net(image4)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(1)))
    return JsonResponse({'result': classes[predicted]})


