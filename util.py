from urllib.request import urlopen

FOTR = 'https://storage.googleapis.com/kagglesdsdata/datasets/45488/83080/01%20-%20The%20Fellowship%20Of%20The%20Ring.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230118%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230118T202629Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b6dea9bf3bbe5b96b58e38c0786a359007f2a5e65e47dfa5583503f0e55b35631022625d09d02b8690108a3d61ca07f7e8665019aca58fca6b66f61a479badea7d64943c02bee3056346446dee4f4f99d1f3dfacb40bbbda49d9afd53a32a3ce799a446c7f694f80904876982f75844fe695ad09ebe59d80157f4db5a95bfbb09a80973e812ed214a4b9fce8cbae03bc7c5916353d24ecf082e8cc794178eebad186f79c8d6ac1bed26a5ad034552bb2659a863b5c3f1d5f1d539dbd4abaa46876274542067e9d781922d92a9a072bf3772257205c6cadcd97c6112ebb87450e977b8f6e8a85a5d591b443a66c1fdeffb8ccfeef83a80333eaa22f550915b774'
TTT = 'https://storage.googleapis.com/kagglesdsdata/datasets/45488/83080/02%20-%20The%20Two%20Towers.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230118%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230118T202943Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=63559951831d82d4bc2595ee6486e7dea9b2a70b4590fa6a36cb69cef55c339c6dd52c41af3ab8c4d15cd01215acd141f5b1fa95c2738ae7c3f8e4c8bbf2bf869a21c83df7b3bb9c46da75c10226029db2bc16b2414d42646e48f60fd2ba3ab3d19ebd53f7ef94890b78894fe338e91956f67d5e85889929c4433dd73f92b85170a012d42f431c1b1160b6824e476a6143b98396a1e7cecffd1d087a88003e4add97435b1c47f655458f34f3a83cb1f9501613dcc1a7a82bd03db67facf4f6aa432de35aa2aa503c8c9c14be7a26efe901df4432cac14b4a6fdab09d3ecb4de7e6e265a6fb1beb07ca1c159445a939908af9dcb7fc07cd9d98278493fbeba56f'
ROTK = 'https://storage.googleapis.com/kagglesdsdata/datasets/45488/83080/03%20-%20The%20Return%20Of%20The%20King.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230118%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230118T203010Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=50df424e1ddef0424d0780e8a0a80bb9adde3ea78387db5f439b69c9352acd7245bf11e8809e466a07be1859a03d250aaa044f1c34601e873646c2feca25ee602804ddf778ef4edabd3ff12d16bf68fcb6f53e58802e369c20bb29d3ce3369862fd7e5c5aea3264e60877ca07a3fe6aa9afaa36ce470c61e6abe78117e45b190a929055722b843203e09a58d1eb5ae54b4c9f24e3f520f864463fcce7f342952d74a7aef21d5c67cbd71833ccd7cc59515ac821c3ec187398eb7410b9b55c072ae7384bb46d6e9dd2595e76c0ac20333d64a637e79fbec910b320b8a8a39e60838d60c424214e6d948b4e5054b42e01e62395f3e32a682d872e109c2feb382ad'

def get_lotr_text():

    save_to_file(FOTR, 'data//FOTR.txt')
    save_to_file(TTT, 'data//TTT.txt')
    save_to_file(ROTK, 'data//ROTK.txt')


def save_to_file(address, filename):
    text = urlopen(address).read()
    with open(filename, 'wb') as f:
        f.write(text)

if __name__ == '__main__':
    get_lotr_text()