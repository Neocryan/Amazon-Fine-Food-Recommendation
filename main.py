# print('choose a dataset: ')
# print('1 will be very slow')
# method0 = input('0 is condense one, 1 is sparse one')
from cf import cf
from cm import drawcm
from MF import MF1
from PMF import PMF
from recommendation import rec
from SVD import svdrec
from calculateMSE import caculate_mse,draw_mse
from rf import rf
print('choose a algorithm')
algo = input('1:MF\n2:PMF\n3:SVD\n4:Distance based\n5:Random Forest\n-->')


print('show confusion matrix?')
cm = input('0 or 1\n-->')

print('do recommandtion?')
rec0 = input('0 or 1\n-->')

if int(algo) == 1:
    mi = input('maxIter? as int\n-->')
    pl = input('plot? 0 or 1\n-->')
    result =MF1( factors=30, maxIter=int(mi), LRate=0.02, GD_end=1e-3, plot=int(pl))
    caculate_mse(result)

    if int(cm) == 1:
        drawcm(result,title='MF')
    if int(rec0) == 1:
        uid = int(input('which user you want to apply(0~3065)\n-->'))
        n = int(input('how many item you want to recommandat, as int\n-->'))
        rec(result, uid,n,rawId= True)

if int(algo) == 2:
    mi = input('maxIter? as int\n-->')
    pl = input('plot? 0 or 1\n-->')
    result =PMF( factors=30, maxIter=int(mi), LRate=0.02, GD_end=1e-3, plot=int(pl))
    caculate_mse(result)

    if int(cm) == 1:
        drawcm(result,title='MF')
    if int(rec0) == 1:
        uid = input('which user you want to apply(0~3065)\n-->')
        uid = int(input('which user you want to apply(0~3065)\n-->'))
        n = int(input('how many item you want to recommandat, as int\n-->'))

if int(algo) == 3:
    fac =int(input('how many factors you want, as int\n-->'))
    result =svdrec(factors=fac)
    caculate_mse(result)

    if int(cm) == 1:
        drawcm(result,title='MF')
    if int(rec0) == 1:
        uid = int(input('which user you want to apply(0~3065)\n-->'))
        n = int(input('how many item you want to recommandat, as int\n-->'))
        rec(result, uid,n,rawId= True)

if int(algo) == 4:
    result =cf()
    caculate_mse(result)

    if int(cm) == 1:
        drawcm(result,title='MF')
    if int(rec0) == 1:
        uid = int(input('which user you want to apply(0~3065)\n-->'))
        n = int(input('how many item you want to recommandat, as int\n-->'))
        rec(result, uid,n,rawId= True)

if int(algo) == 5:
    a = rf()


