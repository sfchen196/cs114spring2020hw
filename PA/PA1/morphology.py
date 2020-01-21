from fst import FST

class Parser():

    def __init__(self):
        pass

    def generate(self, analysis):
        """Generate the morphologically correct word 

        e.g.
        p = Parser()
        analysis = ['p','a','n','i','c','+past form']
        p.generate(analysis) 
        ---> 'panicked'
        """

        # Let's define our first FST
        f1 = FST('morphology-generate')

        output = ['p','a','n','i','c','k','e','d']
        return ''.join(output)

    def parse(self, word):
        """Parse a word morphologically 

        e.g.
        p = Parser()
        word = ['p','a','n','i','c','k','i','n','g']
        p.parse(word)
        ---> 'panic+present participle form'
        """

        # Ok so now let's do the second FST
        f2 = FST('morphology-parse')

        output = ['p','a','n','i','c','+present participle form']
        return ''.join(output)
