from wtforms import Form, SubmitField


class translateHDForm(Form):
    translateHD = SubmitField('Translate with pix2pixHD')

class translateSpadeForm(Form):
    translateSpade = SubmitField('Translate with Spade')

class sampleForm(Form):
    sample = SubmitField('Next Random Sample')