import csv
import re

test = "Test for 3x5 k=3:"

testre = re.compile('Test for (\d)x(\d) k=(\d):')
m = testre.match(test)
meanre = re.compile('Mean:((\d)*.(\d)*)')
sterrorre = re.compile('Standard Error: ((\d)*.(\d)*)')
for group in range(1,4):
    print(m.group(group))

with open('k<=m<=5.txt', 'r') as f:
    data = f.readlines()
for i in range(len(data)):
    params = testre.match(data[i])
    if params:
        m = meanre.match(data[i+5])
        st = sterrorre.match(data[i+6])
        print(params.group(1), params.group(2), params.group(3), m.group(1), st.group(1))

print(data[8])
m = meanre.match(data[8])
st = sterrorre.match(data[9])
print(m)
print(m.group(1))
print(st.group(1))
print(data[9])

def results_to_csv(read_path: str, write_path: str):
    paramre = re.compile('Test for (\d*)x(\d*) k=(\d*):')
    meanre = re.compile('Mean:(((\d)*.(\d)*)e*\+*(\d)*)')
    sterrorre = re.compile('Standard Error: (((\d)*.(\d)*)e*\+*(\d)*)')
    tests = []
    with open(read_path, 'r') as f:
        data = f.readlines()
    for i in range(len(data)):
        params = paramre.match(data[i])
        if params:
            m_nonterm = meanre.match(data[i + 5])
            st_nonterm = sterrorre.match(data[i + 6])
            mn_product = float(params.group(1)) * float(params.group(2))
            values = [params.group(1), params.group(2),params.group(3), str(mn_product), m_nonterm.group(1), st_nonterm.group(1)]
            m_term = meanre.match(data[i + 10])
            values.append(m_term.group(1))
            st_term = sterrorre.match(data[i + 11])
            values.append(st_term.group(1))
            m_illegal = meanre.match(data[i + 15])
            values.append(m_illegal.group(1))
            st_illegal = sterrorre.match(data[i + 16])
            values.append(st_illegal.group(1))
            tests.append(values)
            print(params.group(1), params.group(2), params.group(3), m_nonterm.group(1), st_nonterm.group(1))
    with open(write_path, 'a') as resultfile:
        resultwriter = csv.writer(resultfile, delimiter=',')
        resultwriter.writerows(tests)
    print(tests)

if __name__ == '__main__':
    with open('resulttable.csv', 'a') as resultfile:
        headerwriter = csv.writer(resultfile, delimiter=',')
        header = ["m", "n", "k", "mn", "nonterm_mean", "nonterm_sterror", "term_mean", "term_sterror", "illegal_mean", "illegal_sterror"]
        headerwriter.writerow(header)

    for i in range(4, 11):
        print(i)
        results_to_csv(f'k<=m<={i}.txt', "resulttable.csv")
    # results_to_csv("squareresults.txt", "squaretable.csv")