# 中药、方剂命名实体识别
[《命名实体识别在中药名词和方剂名词识别中的应用》](http://zgys.cnjournals.org/ch/reader/view_abstract.aspx?file_no=20190616&flag=1) 中所使用的数据。

## 标记说明

使用BIO标记方法。中药名词用大写B、I标注，方剂名词用小写b、i标注。其余皆用O标注。

例1：\
黄芪的使用比例决定了药效。\
BIOOOOOOOOOOO


例2：\
麻黄汤药味虽少，但发汗力强，不可过服。\
biiOOOOOOOOOOOOOOOO

目录dictionaries中包含中医学科词典，包括中药、方剂、病症、书籍、针灸等。

## 文章引用

```
@article{deshan_2019_NER,
  title={命名实体识别在中药名词和方剂名词识别中的应用},
  author={龚德山 and 梁文昱 and 张冰珠 and 马星光},
  journal={中国药事},
  volume={33},
  number={6},
  pages={710--716},
  year={2019}
}
```
