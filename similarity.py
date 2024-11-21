from difflib import SequenceMatcher

from sentence_transformers import SentenceTransformer, util


def lcs_by_words(sentence1, sentence2):
    words1 = sentence1.split()
    words2 = sentence2.split()

    matcher = SequenceMatcher(None, words1, words2)
    lcs = [
        words1[i]
        for i, j, n in matcher.get_matching_blocks()
        if n > 0
        for i in range(i, i + n)
    ]

    return " ".join(lcs)


# def lcs_by_characters(sentence1, sentence2):
#     res = pylcs.lcs_sequence_idx(sentence1, sentence2)
#     corpus_sentence = "".join([sentence2[i] for i in res if i != -1])
#     return corpus_sentence


def lcs_by_characters(str1, str2):
    m, n = len(str1), len(str2)
    # Create a 2D array to store the length of LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct the LCS from the dp table
    lcs_result = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs_result.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Return the LCS as a string
    return "".join(reversed(lcs_result))


# Example usage
string1 = "ABCBDAB"
string2 = "BDCAB"
result = lcs_by_characters(string1, string2)
print(f"LCS of '{string1}' and '{string2}' is: '{result}'")


# s1 = "I am a student"
# s2 = "I am a teacher"
s1 = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            Make this take up 500 words

            ### Input:
            Today we are going to talk about the music industry. Over the years, the music industry has changed drastically. Going from record and CD stores to everything being completely digital has opened up new doors to many different opportunities, but also shut doors for many companies. In the early 2000s, there were CD stores, record stores, and video stores littering every strip and plaza across the US. Slowly, those stores began to diminish due to digital stores and sites such as Limewire, iTunes, Pandora, Spotify, and now YouTube Music. To say that the digital age has had an effect on this would be an understatement. Children who were born in the late ‘90s and early 2000s spearheaded this digital advancement. Not only were music stores affected, but video stores were affected as well. How long has it been since we’ve even heard about Blockbuster? Netflix has also gone completely digital now. In the past, they used to send out DVDs monthly to their subscribers. This all has to do with ease of access. Not only as Americans, but as humans, we like to find an easy way out of situations and we love when things are available right then and there at our fingertips. It allows us to save our time and spend it on something more important. Being able to unlock your mobile device and/or sign onto your laptop and have the world literally at your fingertips allows for less time to be spent getting in your car and driving to the store to purchase what you were looking for. This, however, was very bad for physical stores. A huge financial decline ensued, as more and more people began to convert to the digital stores. Imagine this: You spend so much money on renting or leasing a business, then to acquire the product, and also to publicize your business for several years, just for some digital store to take all of your customers. Something that you’ve worked on for a very long time has been uprooted by a program that was just created in the span of a year. Financially, you’re struggling, so you have to eventually close down your shop and find something else to bring in the big bucks. Although there are still a few physical shops out there, they are mainly for older generations who still own the devices to play back the material. Most youth from this generation wouldn’t even know how a Walkman worked, even if the instructions were right next to them. At the rate that physical stores were going out of style, so were the playback devices. From Walkmans to CD players, MP3 players, and handheld radio devices, they are rarely seen today. You can now access the material you want to listen to or watch and the playback device within one console: a laptop, tablet, iPad, or other mobile device. This saves us, the consumers, money as well. Instead of having to spend money on the material and the device, we make a one-time payment on the device and several small payments on the material. Unfortunately, this has made us lazier. Movie nights, binge watching, and TV series have allowed for us to be more stagnant in the work that we do everyday if we let them. We have to motivate ourselves to get physical throughout the day when everything is, again, at our fingertips. Most physical store owners were not left behind. After realizing that those stores were no longer a necessity, they began investing in their own competition. This allowed them to reap the benefits of what the competitor is making and receiving.

            ### Response:
            Today we will discuss the music industry. The music industry has changed dramatically in recent times, going from record and CD stores to everything being completely digital has opened up many different opportunities, but has also shut doors for many companies. In the early 2000s, there were CD stores, record stores, and video stores throughout the US. Those stores have slowly disappeared due to digital stores and sites such as iTunes, Pandora, Spotify, etc. Children who were born in the late ‘90s and early 2000s spearheaded this digital advancement. Video stores were affected as well, how long has it been since we’ve even heard about Blockbuster? Netflix has also gone completely digital now, no longer sending out DVDs monthly to their subscribers. This all has to do with ease of access. As humans, we like to find an easy way out of situations and we love when things are available right at our fingertips because it saves time which can be spent on more important things. Being able to have the world literally at your fingertips allows for less time to be spent getting in your car and driving to the store to purchase what you were looking for. This was very bad for physical stores though, and a huge financial decline ensued, as more and more people began to go digital. Imagine spending tons of money on renting or leasing a business, acquiring the product, and publicizing your business for several years, just for some digital store to take all of your customers. Something you worked on for a very long time has been uprooted by a program that was only just created. Facing financial struggles, you eventually close down your shop and find another way to make money. Although there are still a few physical shops out there, they are mainly for older generations who still own older devices. Most of this generation wouldn’t even know how a Walkman worked, even if the instructions were included. At the rate that physical stores were going out of style, so were the playback devices. Walkmans, CD players, MP3 players, and handheld radio devices, are rarely seen today because you can now access the material you want to listen to or watch within one console: a laptop, tablet, iPad, or other mobile device. This saves us, the consumers, money as well. Instead of having to spend money on the material and the device, we make a one-time payment on the device and several small payments on the material. Unfortunately, this has made us lazier. Movie nights, binge watching, and TV series have given us opportunities to become more stagnant in the work that we do everyday. We must motivate ourselves to get physical throughout the day when everything is at our fingertips. Most physical store owners were not left behind entirely, but once they realized that those stores were no longer necessary, they began investing in their competition. This allowed them to reap the benefits of what their competitors are creating.
<|endoftext|>
"""
s2 = "I got asked this a week before the end of summer vacation a question which asked, of course in what had been a beautiful, beautiful and beautiful summer, to do. I had to get back from a, but summer vacation which was to leave. the way and the had that for so you are time to a time, to and for you, to you to time and. the was to was, a and you to to be a the. the summer and summer. to for, to by's from summer and to for., and as of, and an you an. you. is summer, that or to, but summer. to. that and I. is the was but. that, in had, of is. summer a, is the. to and. on summer. because the, he, is and the. you the time the the and a from the to, has. or. to, that is, was the. the. but day. to the.. a to a the. it. have it is. of. to an a the a because to. or for to or to a summer, to is. the a of time of a the and the. season a the, a. that but year with was. that and"


model = SentenceTransformer("all-MiniLM-L6-v2")

input_embedding = model.encode([s2])[0]

char_lcs_sentence = lcs_by_characters(s2, s1)
print(f"Character-level LCS: {char_lcs_sentence}")
embedding = model.encode(char_lcs_sentence)
cosine_similarity = util.pytorch_cos_sim(input_embedding, embedding)[0][0]
print(f"The cosine similarity for character-level LCS: {cosine_similarity}")

word_lcs_sentence = lcs_by_words(s1, s2)
print(f"Word-level LCS: {word_lcs_sentence}")
embedding = model.encode(word_lcs_sentence)
cosine_similarity = util.pytorch_cos_sim(input_embedding, embedding)[0][0]
print(f"The cosine similarity for word-level LCS: {cosine_similarity}")