import scipy.spatial.distance as ssd
from config import *
from get_input import loadData


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def loadDataDirectTest(mode, shuffleList, batchIndex):
    batchIndex = shuffleList[batchIndex]
    if mode == "sketch":

        image_list, label_list, bbox_list, img, adj, corr = loadData(
            os.path.join(sketchVPathTest, str(batchIndex).zfill(12) + ".csv"),
            os.path.join(sketchImgTestPath, str(batchIndex).zfill(12) + ".jpg"))
    else:
        image_list, label_list, bbox_list, img, adj, corr = loadData(
            os.path.join(imageVPathTest, str(batchIndex).zfill(12) + ".csv"),
            os.path.join(imageImgTestPath, str(batchIndex).zfill(12) + ".jpg"))

    return image_list, label_list, bbox_list, img, adj, corr


def outputHtml(sketchindex, indexList):
    imageNameList = shuffleListTest
    sketchPath = sketchImgTestPath
    imgPath = imageImgTestPath

    tmpLine = "<tr>"

    tmpLine += "<td><image src='%s' width=256 /></td>" % (
        os.path.join(sketchPath, str(shuffleListTest[sketchindex]).zfill(12) + ".png"))
    for i in indexList:
        if i != sketchindex:
            tmpLine += "<td><image src='%s' width=256 /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))
        else:
            tmpLine += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join(imgPath, str(imageNameList[i]).zfill(12) + ".jpg"))

    return tmpLine + "</tr>"


def calculate_accuracy(dist, epoch_name):
    top1 = 0
    top5 = 0
    top10 = 0
    top20 = 0
    tmpLine = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        if i in rank[:20]:
            top20 = top20 + 1
        tmpLine += outputHtml(i, rank[:10]) + "\n"
    num = dist.shape[0]
    print(epoch_name + ' top1: ' + str(top1 / float(num)))
    print(epoch_name + ' top5: ' + str(top5 / float(num)))
    print(epoch_name + 'top10: ' + str(top10 / float(num)))
    print(epoch_name + 'top20: ' + str(top20 / float(num)))

    htmlContent = """
       <html>
       <head></head>
       <body>
       <table>%s</table>
       </body>
       </html>""" % (tmpLine)
    with open(r"html_result/result.html", 'w+') as f:
        f.write(htmlContent)
    return top1, top5, top10, top20
