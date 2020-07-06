import re


def remove_source(text):
    cln_text = text
    if '(Source' in cln_text:
        cln_text,_,_ = cln_text.partition('(Source')
    elif '[Written ' in cln_text:
        cln_text,_,_ = cln_text.partition('[Written')
        
    return cln_text

def clean_synopsis(data):
    # removing hentai and kids tags
    data = data[(data.Hentai != 1) & (data.Kids != 1)]
    synopsis = data.synopsis
    
    # removing very small synopsis
    synopsis = synopsis.apply(lambda x: x if ((len(str(x).strip().split())<=300) and len(str(x).strip().split())>30  ) else -1)
    synopsis = synopsis[synopsis!=-1]
    
    # removing source text
    synopsis = synopsis.apply(lambda x: remove_source(x))
    
    # removing japanese characters
    synopsis = synopsis.apply(lambda x: re.sub("([^\x00-\x7F])+"," ",x))
    
    # remove symbols
    rx = re.compile('[&#/@`)(;<=\'"$%>]')
    synopsis = synopsis.apply(lambda x: rx.sub('',x))
    synopsis = synopsis.apply(lambda x: x.replace('>',""))
    synopsis = synopsis.apply(lambda x: x.replace('`',""))
    synopsis = synopsis.apply(lambda x: x.replace(')',""))
    synopsis = synopsis.apply(lambda x: x.replace('(',""))
    

    # removing adaptation animes (some relevant might get deleted but there aren`t a lot so we wont be affected as much)
    synopsis = synopsis[synopsis.apply(lambda x: 'adaptation' not in str(x).lower())]    
    synopsis = synopsis[synopsis.apply(lambda x: 'music video' not in str(x).lower())]
    synopsis = synopsis[synopsis.apply(lambda x: 'based on' not in str(x).lower())]
    synopsis = synopsis[synopsis.apply(lambda x: 'spin-off' not in str(x).lower())]
    
    return synopsis.reset_index(drop=True)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
