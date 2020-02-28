
from mongoengine import fields
from mongoengine.document import Document

class URL(Document):
    meta = {'strict': False, 'collection': 'urls'}
    url = fields.URLField(required=True, unique=True)
    title = fields.StringField()
    lang = fields.StringField()
    imageUrls = fields.ListField(fields.StringField())
    words = fields.ListField(fields.StringField())
    words_len = fields.IntField()
    lastmod = fields.StringField()
    requestId = fields.StringField()
    passage = fields.StringField()
    user_meta = fields.DictField()
