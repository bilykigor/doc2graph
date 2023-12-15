import re
from nltk.tokenize import word_tokenize
from price_parser import Price
import dateparser



def get_mask_tokens(text, mask, min_length=0):
    "find tokens in text that match with mask"
    candidates = list(
        {
                token
                for token in word_tokenize(text.lower())
                if re.match(mask, token)
                and re.match(mask, token).span() == (0, len(token))
        }
    )

    candidates = [x for x in candidates if len(x)>min_length]

    return candidates


def find_amounts(text):
    text = text.lower().strip()
    currencies = r'(?:\$|€|£|¥|eur|usd|gbp|\u20AC|s|S)?'
    mask_1 = r'(([1-9]\d+|0)[,.]\d{1,3})' #0,12123; 0.321233
    mask_2 = r'[1-9]\d{0,2}(?:([,])\d{3})?(?:\1\d{3})*([.]\d{1,3})' #1,321,233.23
    mask_3 = r'[1-9]\d{0,2}(?:([.])\d{3})?(?:\1\d{3})*([,]\d{1,3})' #1,321,233.23
    amount_candidates_1 = set(get_mask_tokens(text, f'{currencies}-?{mask_1}'))
    amount_candidates_2 = set(get_mask_tokens(text, f'{currencies}-?{mask_2}'))
    amount_candidates_3 = set(get_mask_tokens(text, f'{currencies}-?{mask_3}'))
    amount_candidates = list(amount_candidates_1.union(amount_candidates_2).union(amount_candidates_3))
    
    if amount_candidates:
        return True
    return False


def find_codes(text):
    # text = text.lower().strip()
    # mask = r"#?([a-z]+-?)*\d+[-a-z\d]*"
    # mt = re.match(mask, text)
    # if mt:
    #     if mt.span() == (0, len(text)):
    #         return True
    # return False
    
    if has_digits(text):
        if has_no_digits(text):
            return True
    
    return False


def is_digit(input_string):
    for char in input_string:
        if char not in '0123456789':
            return False
    return True


def has_digits(input_string):
    for char in input_string:
        if char in '0123456789':
            return True
    return False


def has_no_digits(input_string):
    for char in input_string:
        if char not in '0123456789':
            return True
    return False


def is_letters(input_string):
    for char in input_string:
        if not char.isalpha():
            return False
    return True


def has_letters(input_string):
    for char in input_string:
        if char.isalpha():
            return True
    return False


def remove_stop_symbols(text):
    if has_digits(text):
        return text
    
    #stop_symbols = '()|$&*#!/-'
    #for s in stop_symbols:
    #    text = text.replace(s,' ')
    
    text = [s if s.isalpha() or s == ' ' else ' ' for s in text]
    
    text = ''.join(text).strip()
    
    return text


def has_only_letters_and_space(text):
    text = remove_stop_symbols(text)
    
    pattern = r'^[A-Za-z\s]+$'
    return re.match(pattern, text) is not None


def has_only_digits_and_space(text):
    text = remove_stop_symbols(text)
    
    pattern = r'^[0-9\s]+$'
    return re.match(pattern, text) is not None


def find_numbers(text):
    text = text.lower().strip()
    
    return text.isdigit()


def find_word(text):
    text = text.lower().strip()
    
    mask = r'[a-zA-Z.,]+'
    if re.match(mask, text) and re.match(mask, text).span() == (0, len(text)):
        return True
    return False
        
        
def find_words(text):
    text = text.lower().strip()
    
    mask = r'[a-zA-Z :_.,&#/@-]+'
    if re.match(mask, text) and re.match(mask, text).span() == (0, len(text)):
        return True
    return False


def find_dates(s):
    s = s.lower().strip()
    chunks = re.findall(r'\d+|[A-Za-z]+|\W+', s)
    if len(chunks)>5:
        return False
        
    digit_chunks = [x for x in chunks if x.isdigit()]
    if len(digit_chunks)>3:
        return False
    
    word_chunks = [x for x in chunks if x.isalpha()]
    if len(word_chunks)>1:
        return False
    
    other_chunks = [x for x in chunks if not (x.isalpha() or x.isdigit())]
    if len(other_chunks)>2:
        return False
    
    for oc in other_chunks:
        if oc.strip() not in ['.','','/','-',',',' ']:
            return False
    
    if word_chunks:
        if word_chunks[0] not in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','january','february','march','april','may','june','july','august','september','october','november','december']:
            return False
        
        if len(digit_chunks)!=2:
            return False
        
        if int(digit_chunks[0]) not in range(1,32):
            return False
        
        if len(digit_chunks[1]) not in [1,2,4]:
            return False
        
        if len(digit_chunks[1])==4:
            if int(digit_chunks[1])<1900:
                return False
            if int(digit_chunks[1])>2030:
                return False
    else:
        if len(digit_chunks)!=3:
            return False
        
        if int(digit_chunks[1]) not in range(1,32):
            return False
        
        second_day = 32
        if int(digit_chunks[1]) in range(13,32):
            second_day = 13
        
        if len(digit_chunks[0]) not in [1,2,4]:
            return False
        
        if len(digit_chunks[2]) not in [1,2,4]:
            return False
        
        if (len(digit_chunks[0])==4) and (len(digit_chunks[2])==4):
            return False
        
        if len(digit_chunks[0]) in [1,2]:
            if int(digit_chunks[0]) not in range(1,second_day):
                return False
        
        if len(digit_chunks[0])==4:
            if int(digit_chunks[0])<1900:
                return False
            if int(digit_chunks[0])>2030:
                return False
            if int(digit_chunks[2]) not in range(1,second_day):
                return False
            
        if len(digit_chunks[2])==4:
            if int(digit_chunks[2])<1900:
                return False
            if int(digit_chunks[2])>2030:
                return False
            
    return True


def text_to_mask(text):
    emb=[0,0,0,0,0,0]
    if find_dates(text):
        #print(text)
        emb[0]=1
    elif find_amounts(text):
        #print(text)
        emb[1]=1
    elif find_numbers(text):
        #print(text)
        emb[2]=1
    elif find_codes(text):
        #print(text)
        emb[3]=1
    elif find_word(text) or find_words(text):
        #print(text)
        emb[4]=1
    #else:
    #    print(text)
    #emb.append(len(text))
    #emb.append(len(text.split()))
    return emb
