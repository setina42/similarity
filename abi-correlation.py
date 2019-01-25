import os
import urllib
import json
import urllib
import urllib.request
from collections import defaultdict
import argparse

def get_correlation(id):
   """Queries the ABI API for an expression summary for structure-id = 997 (the entire brain) and returns
   expression density, expression energy. Used to filter out datasets that barely measure any expression.
"""

   results = defaultdict(list)
   url = "http://api.brain-map.org/api/v2/data/query.json?criteria=service::mouse_correlation[set$eqmouse][row$eq{}]".format(str(id))
   print(url)
   source = urllib.request.urlopen(url).read()
   response = json.loads(source)
   
   for x in response['msg']:
      print("new x")
      print(x)
      gene_id = x['id']
      gene_acronym = x['gene-symbol']
      gene_r = x['r']
      
      results[gene_id].append(gene_acronym)
      results[gene_id].append(gene_r)
   
   return results,id



def output_results(results,acronym,id):
   file_path="ABI-correlation-{0}-{1}.csv".format(acronym,id)
   f = open(file_path,"w+")
   for id in results:
        f.write('\n')
        f.write(str(id))
        for bla in results[id]:
            f.write("," + str(bla))
   f.close()


def main():
   parser = argparse.ArgumentParser(description="ABI-expression",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('--id','-i',type=int)
   parser.add_argument('--acronym','-a')
   args = parser.parse_args()
   
   results,id = get_correlation(args.id)
   output_results(results,args.acronym,id)

if __name__ == "__main__":
    main()
