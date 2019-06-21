'''
Created on 2009-10-30

@author: Administrator
'''
from random import random, randint, choice, uniform
from copy import deepcopy
from PIL import Image, ImageDraw
from math import sqrt, pi
from numpy import argsort, argmax, array
from scipy.optimize import minimize
import time


class funwrapper:
  def __init__(self, function, childcount, name):
    self.function = function
    self.childcount = childcount
    self.name = name
  def getchildcount(self):
    return self.childcount
  def getname(self):
    return self.name

class variable:
  def __init__(self, var, value=0):
    self.var = var
    self.value = value
    self.name = str(var)
    self.type = "variable"

  def getname(self):
    return self.var

  def evaluate(self):
    return self.varvalue

  def setvar(self, value):
    self.value = value

  def display(self, indent=0):
    print ('%s%s' % (' '*indent, self.var))

class const:
  def __init__(self, value):
    self.value = value
    self.name = str(value)
    self.type = "constant"

  def setvalue(self,newvalue):
    self.value=newvalue
    self.name=str(newvalue)
  def getname(self):
    return self.name

  def evaluate(self):
    return self.value

  def display(self, indent=0):
    print ('%s%d' % (' '*indent, self.value))

class node:
  def __init__(self, nodedepth,numoffun,type, children, funwrap, var=None, const=None, parent=None ):
    self.type = type
    self.children = children
    self.funwrap = funwrap
    self.variable = var
    self.const = const
    self.depth = self.refreshdepth()
    self.value = 0
    self.fitness = 0
    self.numoffun=numoffun
    self.nodedepth=nodedepth
    self.parent=parent

  def get_depth(self):
    return self.depth

  def getnodedepth(self):
    print(self.nodedepth)

  def getNodesFromLayer(self, selecteddepth):

    if self.type=="function":
      if self.nodedepth == selecteddepth:
        #print("nodedepth==selecteddepth")
        return [self]
      else:
        nodes = []
        for c in self.children:
          nodes += c.getNodesFromLayer(selecteddepth)
        if nodes:
          return nodes
        else:
          return []
    else:
      return []


  def getSimilarNodes(self,other):
    if self.type==other.type:
      if self.type=="function" and self.funwrap.getchildcount() == other.funwrap.getchildcount():
        nodes = [[self, other]]
        offspring_nodes = []
        for self_c, other_c in zip(self.children, other.children):
          tmp = self_c.getSimilarNodes(other_c)
          if tmp:
            if type(tmp[0]) is list:
              for i in tmp:
                offspring_nodes.append(i)
            else:
              offspring_nodes.append(tmp)
        if offspring_nodes:
          nodes += offspring_nodes
        return nodes
      elif self.type != "function" and self.type == other.type:
        return [self, other]
      else:
        return
    else:
      return

  def get_constant_nodes(self):
    if self is None: return

    arch = []
    if self.type=="function":
      for node in self.children[:-1]:
        offspring_layers = node.get_constant_nodes()
        for layer in offspring_layers:
          arch.append(layer)

      offspring_layers = self.children[-1].get_constant_nodes()
      for layer in offspring_layers:
        arch.append(layer)

      return arch
    elif self.type=="constant":
      return [self]
    else:
      return []


  def getParentsfornodes(self, parentfromoutside=None):
    self.parent=parentfromoutside
    if self.type == "function":
      for c in self.children:
        c.getParentsfornodes(parentfromoutside=self)


  def refreshdepthAfterCrossover(self,setnodedepth):

    self.depth = self.refreshdepth()
    self.nodedepth = setnodedepth
    if self.type=="function":
      for c in self.children:
        c.refreshdepthAfterCrossover(setnodedepth+1)

  def nodemutate (self,probability, funwraplist, variablelist,constantlist):

    #self.drawtree("nodemutate.png")
    if self.type == "function":
      if random() < probability:
        selectedfun = randint(0, len(funwraplist) - 1)
        while funwraplist[selectedfun].getchildcount() != self.funwrap.getchildcount():
          selectedfun = randint(0, len(funwraplist) - 1)
        self.numoffun = selectedfun
        self.funwrap = funwraplist[selectedfun]
      #self.drawtree("nodemutate.png")
      for c in self.children:
        c.nodemutate(probability, funwraplist, variablelist,constantlist)
    if self.type == "variable":
      if random() < probability:
        selectedvariable = randint(0, len(variablelist) - 1)
        self.variable = variable(variablelist[selectedvariable])
        #print("selfvariable")
        #print(self.variable)
        #self.drawtree("nodemutate.png")
    if self.type == "constant":
      if random() < probability:
        selectedconstant = randint(0, len(constantlist) - 1)
        self.const = const(constantlist[selectedconstant])
        #print("selfconst")
        #print(self.const)
        #self.drawtree("nodemutate.png")


  def eval(self):
    if self.type == "variable":
      return self.variable.value
    elif self.type == "constant":
      return self.const.value
    else:
      if self.funwrap.getchildcount() == 1:
        try:

          return self.funwrap.function(self.children[0].eval())
        except OverflowError:
          return float('inf')
      else:
        try:

          return self.funwrap.function(self.children[0].eval(),self.children[1].eval())
        except:
          return float('inf')



  def getfitness(self, checkdata, minindata, maxindata):#checkdata like {"x":1,"result":3"}
    diff = 0
    #set variable value
    for data in checkdata:
      self.setvariablevalue(data)
      diff += (self.eval() - data["result"])**2

    diff=diff/len(checkdata)
    diff=sqrt(diff)
    diff=diff/(maxindata-minindata)
    diff=1.0/(1.0+diff)
    self.fitness = diff

  def geterror_of_besttree(self,checkdata, minindata, maxindata):#checkdata like {"x":1,"result":3"}
    diff = 0

    for data in checkdata:
      self.setvariablevalue(data)
      diff += (self.eval() - data["result"])**2

    diff=diff/len(checkdata)
    diff=sqrt(diff)
    diff=diff/(maxindata-minindata)
    return diff

  def setvariablevalue(self, value):
    if self.type == "variable":
      if value.has_key(self.variable.var):
        self.variable.setvar(value[self.variable.var])
      else:
        print ("There is no value for variable:", self.variable.var)
        return
    if self.type == "constant":
      pass
    if self.children:#function node
      for child in self.children:
        child.setvariablevalue(value)

  def refreshdepth(self):
    if self.type == "constant" or self.type == "variable":
      return 0
    else:
      depth = []
      for c in self.children:
        depth.append(c.refreshdepth())
      return max(depth) + 1

  def __cmp__(self, other):
        return cmp(self.fitness, other.fitness)

  def display(self, indent=0):
    if self.type == "function":
      print ('  '*indent) + self.funwrap.name
    elif self.type == "variable":
      print ('  '*indent) + self.variable.name
    elif self.type == "constant":
      print ('  '*indent) + self.const.name
    if self.children:
      for c in self.children:
        c.display(indent + 1)
  ##for draw node
  def getwidth(self):
    if self.type == "variable" or self.type == "constant":
      return 1
    else:
      result = 0
      for i in range(0, len(self.children)):
        result += self.children[i].getwidth()
      return result
  def drawnode(self, draw, x, y):
    if self.type == "function":
      allwidth = 0
      for c in self.children:
        allwidth += c.getwidth()*100
      left = x - allwidth / 2
      #draw the function name
      draw.text((x - 10, y - 10), self.funwrap.name, (0, 0, 0))

      #draw the children
      for c in self.children:
        wide = c.getwidth()*100
        draw.line((x, y, left + wide / 2, y + 100), fill=(255, 0, 0))
        c.drawnode(draw, left + wide / 2, y + 100)
        left = left + wide
    elif self.type == "variable":
      draw.text((x - 5 , y), self.variable.name, (0, 0, 0))
    elif self.type == "constant":
      draw.text((x - 5 , y), self.const.name, (0, 0, 0))

  def drawtree(self, jpeg="tree.png"):
    w = self.getwidth()*100
    h = self.depth * 100 + 120

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    self.drawnode(draw, w / 2, 20)
    img.save(jpeg, 'PNG')




class enviroment:
  def __init__(self, funwraplist, variablelist, constantlist, checkdata, checkdataforbesttree,percent,\
               minimaxtype="max", population=None, size=100, maxdepth=10
               , \
               maxgen=300, crossrate=0.9, mutationrate=0.1, newbirthrate=0.6, typeofsel="tournament_sel_9", tour=9, crosstype="onepoint",selfconfig=True,mutatetype="growth"):
    self.funwraplist = funwraplist
    self.variablelist = variablelist
    self.constantlist = constantlist
    self.checkdata = checkdata
    self.minimaxtype = minimaxtype
    self.maxdepth = maxdepth
    self.population = population or self._makepopulation(size)
    self.size = size
    self.maxgen = maxgen
    self.crossrate = crossrate
    self.mutationrate = mutationrate
    self.newbirthrate = newbirthrate
    self.typeofsel=typeofsel
    self.tour=tour
    self.crosstype=crosstype
    self.selfconfig=selfconfig
    self.mutatetype=mutatetype
    self.checkdataforbesttree=checkdataforbesttree
    self.percent=percent

    # Finding min and max for the fitness function
    self.resultlist = [self.checkdata[i]["result"] for i in range(0, len(self.checkdata))]
    self.minindata = min(self.resultlist)
    self.maxindata = max(self.resultlist)

    self.besttree = self.population[0]
    for i in range(0, self.size):
      self.population[i].depth=self.population[i].refreshdepth()
      self.population[i].getfitness(self.checkdata, self.minindata, self.maxindata)
      self.population[i].getParentsfornodes()

      if random()<0.8:
        self.Tree_optimize(self.population[i], self.checkdata, self.minindata, self.maxindata)

      self.population[i].getfitness(self.checkdata, self.minindata, self.maxindata)



      if self.minimaxtype == "min":
        if self.population[i].fitness < self.besttree.fitness:
          self.besttree = self.population[i]
      elif self.minimaxtype == "max":
        if self.population[i].fitness > self.besttree.fitness:
          self.besttree = self.population[i]





  def _makepopulation(self, popsize):
    return [self._maketree(0) for i in range(0, popsize)]

  def _maketree(self, startdepth,maxdepth1=0):
    maxdepthinsidethisF=0
    if maxdepth1==0:
      maxdepthinsidethisF=self.maxdepth
    else:
      maxdepthinsidethisF=maxdepth1


    if startdepth == 0:
      #make a new tree
      nodepattern = 0#function
    #elif startdepth == self.maxdepth:
    elif startdepth == maxdepthinsidethisF:
      nodepattern = 1#variable or constant
    else:
      nodepattern = randint(0, 1)
    if nodepattern == 0:
      childlist = []
      selectedfun = randint(0, len(self.funwraplist) - 1)
      #Tree creating - for each node with function
      for i in range(0, self.funwraplist[selectedfun].childcount):

        child = self._maketree(startdepth + 1,maxdepth1)
        childlist.append(child)


      return node(startdepth,selectedfun,"function", childlist, self.funwraplist[selectedfun])
    else:

      if randint(0, 1) == 0:#variable
        selectedvariable = randint(0, len(self.variablelist) - 1)
        return node(startdepth,None,"variable", None, None, \
               variable(self.variablelist[selectedvariable]), None)

      else:
        selectedconstant = randint(0, len(self.constantlist) - 1)

        return node(startdepth, None, "constant", None, None, None, \
               const(self.constantlist[selectedconstant]))

  def res_func(self, x, tree,constant_nodes , checkdata,minindata,maxindata):

    for xi, node in zip(x, constant_nodes):
      node.const.value = xi
    error=tree.geterror_of_besttree(checkdata, minindata, maxindata)

    return error


  def Tree_optimize(self, tree,checkdata, minindata, maxindata):#checkdata like {"x":1,"result":3"}
      maxiter=10
      copy_tree = deepcopy(tree)
      constant_nodes = copy_tree.get_constant_nodes()

      if constant_nodes:
        x0 = array([coef_node.const.evaluate() for coef_node in constant_nodes])


        try:
          res = minimize(self.res_func, x0, args=(copy_tree, constant_nodes,checkdata,minindata,maxindata), method='L-BFGS-B',
                         options={'maxiter': maxiter, 'disp': False})


        except:
          print("Errof of coefficient's optimization")
        else:
          constant_nodes_of_tree=tree.get_constant_nodes()
          for xi, node in zip(res.x, constant_nodes_of_tree):

            node.const.setvalue(xi)



  def mutate(self, tree, probability=None , startdepth=0, mutation_type="growth"):

    result = deepcopy(tree)

    if not probability:
      if mutation_type == "weak":
        probability = 1.0 / (5.0 * tree.depth)
      elif mutation_type is "standard":
        probability = 1.0 / tree.depth
      elif mutation_type is "strong":
        probability = 5.0 / tree.depth

    if mutation_type == "growth":

      probability_ofgrowth_mutate=0.8

      if random()<probability_ofgrowth_mutate: #probability of growth mutation
        rnlayer = randint(0, tree.depth-1)




        rnnode = choice(tree.getNodesFromLayer(rnlayer))

        rnnode_copy=deepcopy(rnnode)


        rnnode_copy=self._maketree(startdepth,maxdepth1=rnnode.depth)



        if rnlayer==0:
          result=deepcopy(rnnode_copy)
          result.refreshdepthAfterCrossover(setnodedepth=0)

        else:
          result.children = []

          for c in tree.children:
            result.children.append(self.crossover(c, rnnode_copy, 0, rnnode))



          result.refreshdepthAfterCrossover(setnodedepth=0)
      else:
        return deepcopy(tree)

      #rnnode_copy.drawtree("rnnode_copy_aftertreemake.png")

    elif mutation_type in ("weak", "standard", "strong"):


      result.nodemutate(probability,self.funwraplist,self.variablelist,self.constantlist)


    return result




  def crossover(self, tree1, tree2, top=1, changednode=None, crossovertype="onepoint"):

    theend=False

    if top:

      result = deepcopy(tree1)

      if crossovertype=="empty":
        return result


      if crossovertype=="standard":

        rnlayer = randint(0, tree1.depth-1)

        rnselflayer = randint(0, tree2.depth-1)




        changednode = choice(tree1.getNodesFromLayer(rnlayer))
        nodeforchange = choice(tree2.getNodesFromLayer(rnselflayer))



      elif crossovertype=="onepoint":

        common_nodes = tree1.getSimilarNodes(tree2)


        if common_nodes:
          if common_nodes[1:]:
            changednode, nodeforchange = choice(common_nodes[1:])


          else:
            return deepcopy(tree2)


        else:
          return result


      if changednode.nodedepth + nodeforchange.depth - nodeforchange.nodedepth < self.maxdepth:

        if tree1 is changednode:
          result=deepcopy(nodeforchange)
          result.refreshdepthAfterCrossover(setnodedepth=0)
          return result
        else:
          result.children = []
          for c in tree1.children:
            result.children.append(self.crossover(c, nodeforchange, 0, changednode,crossovertype))
          theend=True

      else:
        return deepcopy(tree1)

    else:

      if tree1 is changednode:
        result = deepcopy(tree2)
        if tree1.type=="function":
          result.children = []
          for c in tree2.children:
            result.children.append(self.crossover(c, tree2, 0, changednode,crossovertype))
        else:
          return deepcopy(tree2)
      else:
        result = deepcopy(tree1)
        if tree1.type == "function":
          result.children = []
          for c in tree1.children:
            result.children.append(self.crossover(c, tree2, 0, changednode,crossovertype))
        else:
          return deepcopy(tree1)


    if theend:
      result.refreshdepthAfterCrossover(setnodedepth=0)

    return result

  #Function for generation types of operators for self-configuring
  def gener(self,mass):
    #print (len(mass))
    ran = randint(0, 10000)
    ran = ran / 10000.0
    sum = 0
    for i,num_of_operator in zip(mass, range(0,len(mass))):
      sum += mass[i]
      if sum > ran:
        return i
      elif num_of_operator==len(mass)-1:
        return i


  def evolution(self, maxgen=300):

    average_num_of_generation=None

    ##################Self-configuration##############################
    from collections import OrderedDict
    if self.selfconfig:
      num_of_operatorstypes=3
      num_of_operators=OrderedDict()
      num_of_operators['number_of_sel_types'] = 5
      num_of_operators['number_of_cross_types']= 3
      num_of_operators['number_of_mutate_types'] = 4

      probabilities=OrderedDict()
      selection_p = OrderedDict()
      crossover_p = OrderedDict()
      mutate_p = OrderedDict()
      selection_p["tournament_sel_2"] = 1.0 / num_of_operators['number_of_sel_types']
      selection_p["tournament_sel_5"] = 1.0 / num_of_operators['number_of_sel_types']
      selection_p["tournament_sel_9"] = 1.0 / num_of_operators['number_of_sel_types']
      selection_p["proportional"] = 1.0 / num_of_operators['number_of_sel_types']
      selection_p["rank"] = 1.0 / num_of_operators['number_of_sel_types']

      crossover_p["empty"] = 0.1
      crossover_p["standard"] = 0.9 / (num_of_operators['number_of_cross_types'] - 1)
      crossover_p["onepoint"] = 0.9 / (num_of_operators['number_of_cross_types'] - 1)

      mutate_p["growth"] = 1.0 / num_of_operators['number_of_mutate_types']
      mutate_p["weak"] = 1.0 / num_of_operators['number_of_mutate_types']
      mutate_p["standard"] = 1.0 / num_of_operators['number_of_mutate_types']
      mutate_p["strong"] = 1.0 / num_of_operators['number_of_mutate_types']

      probabilities['selection_probabilities']=selection_p
      probabilities['crossover_probabilities'] = crossover_p
      probabilities['mutate_probabilities'] = mutate_p


      # Threshold probabilities
      Pij=OrderedDict()
      Pij["Pij_s"] = 3.0 / (10.0 * num_of_operators['number_of_sel_types'])
      Pij["Pij_c"] = 3.0 / (10.0 * num_of_operators['number_of_cross_types'])
      Pij["Pij_m"] = 3.0 / (10.0 * num_of_operators['number_of_mutate_types'])

      operators = [[] for k in range(0, num_of_operatorstypes)]



    ##########################################################################

    for i in range(0, maxgen):

      print("generation num")
      print(i)
      ########################Self-configuration part##############################
      if self.selfconfig:


        if i != 0:

          taken_away_from_selection = 0
          taken_away_from_crossover = 0
          taken_away_from_mutation = 0
          for k, k1, k2 in zip(probabilities, Pij, num_of_operators):
            for kk in probabilities[k]:
              if probabilities[k][kk] == Pij[k1]:
                pass
              elif probabilities[k][kk] < Pij[k1] + (1.0 / (num_of_operators[k2] * maxgen)) and probabilities[k][kk] < Pij[k1]:
                probabilities[k][kk] = Pij[k1]
              elif probabilities[k][kk] > Pij[k1] + (1.0 / (num_of_operators[k2] * maxgen)):
                probabilities[k][kk] = probabilities[k][kk] - (1.0 / (num_of_operators[k2] * maxgen))
                if k1 == "Pij_s":
                  taken_away_from_selection += (1.0 / (num_of_operators[k2] * maxgen))
                elif k1 == "Pij_c":
                  taken_away_from_crossover += (1.0 / (num_of_operators[k2] * maxgen))
                elif k1 == "Pij_m":
                  taken_away_from_mutation += (1.0 / (num_of_operators[k2] * maxgen))


          best_operators_types=[] #sel cross mutate


          for o, geneticoperatortype in zip(probabilities, range(0, num_of_operatorstypes)):
            max_average_fitness = 0
            for o1, o2 in zip(probabilities[o], range(0, len(probabilities[o]))):
              #for k in range(0, len(operators)):
                sum = 0
                average_fit = 0
                for k1,k2 in zip(operators[geneticoperatortype],range(0,self.size-1)):
                  if k1 == o1:
                    average_fit+=self.population[k2].fitness
                    sum += 1
                if o2 == 0:
                  if sum != 0:
                    max_average_fitness = average_fit / sum
                  best_operators_types.append(o1)
                else:
                  if sum != 0:
                    if average_fit / sum > max_average_fitness:
                      max_average_fitness = average_fit / sum
                      best_operators_types[geneticoperatortype] = o1

          #print ("best_operators_types")
          #print (best_operators_types)


          probabilities['selection_probabilities'][best_operators_types[0]]+=taken_away_from_selection
          probabilities['crossover_probabilities'][best_operators_types[1]]+=taken_away_from_crossover
          probabilities['mutate_probabilities'][best_operators_types[2]]+=taken_away_from_mutation





      child = []
      for j in range(0, self.size):

        #clon creating
        if j==self.size-1:
          besttree_copy=deepcopy(self.besttree)
          child.append(besttree_copy)
        else:



          if self.selfconfig:
            #selection type

            self.typeofsel = self.gener(selection_p)
            operators[0].append(self.typeofsel)
            self.crosstype=self.gener(crossover_p)
            operators[1].append(self.crosstype)
            self.mutatetype=self.gener(mutate_p)
            operators[2].append(self.mutatetype)



          parent1, p1 = self.selection(self.typeofsel)
          parent2, p2 = self.selection(self.typeofsel)

          while p1==p2:
            parent2, p2 = self.selection(self.typeofsel)



          newchild = self.crossover(parent1, parent2,crossovertype=self.crosstype)


          #newchild.drawtree("result.png")



          newchildaftermutation = self.mutate(newchild,mutation_type=self.mutatetype)

          child.append(newchildaftermutation)


      for j in range(0, self.size):
        self.population[j] = child[j]

      for k in range(0, self.size):
        if random() < 0.8:
          self.Tree_optimize(self.population[k], self.checkdata, self.minindata, self.maxindata)
        self.population[k].getfitness(self.checkdata, self.minindata, self.maxindata)
        self.population[k].depth=self.population[k].refreshdepth()
        if self.minimaxtype == "min":
          if self.population[k].fitness < self.besttree.fitness:
            self.besttree = self.population[k]
        elif self.minimaxtype == "max":
          if self.population[k].fitness > self.besttree.fitness:
            self.besttree = self.population[k]
      print ("best tree's fitness..",self.besttree.fitness)

      if self.besttree.fitness>=0.99:
        average_num_of_generation=i
        with open('out.txt', 'w') as out:
          for key in probabilities:
            out.write('{}\n'.format(probabilities[key]))
        self.besttree.display()
        self.besttree.drawtree()
        break




    res=self.besttree.geterror_of_besttree(self.checkdata, self.minindata, self.maxindata)






    values=[res,average_num_of_generation]
    print ("values")
    print(values)

    return values

  def gettoptree(self, choosebest=0.9, reverse=False):
    if self.minimaxtype == "min":
      self.population.sort()
    elif self.minimaxtype == "max":
      self.population.sort(reverse=True)

    if reverse == False:
      if random() < choosebest:
        i = randint(0, self.size * self.newbirthrate)
        return self.population[i], i
      else:
        i = randint(self.size * self.newbirthrate, self.size - 1)
        return self.population[i], i
    else:
      if random() < choosebest:
        i = self.size - randint(0, self.size * self.newbirthrate) - 1
        return self.population[i], i
      else:
        i = self.size - randint(self.size * self.newbirthrate,\
            self.size - 1)
        return self.population[i], i

  def selection(self, typeofsel):
    #proportional selection
    if typeofsel=="proportional":
      allfitness = 0
      for i in range(0, self.size):
        allfitness += self.population[i].fitness
      #randomnum = random()*(self.size - 1)
      randomnum = randint(0,10000)
      randomnum=randomnum/10000.0
      #print(randomnum)
      check = 0
      #check1=0
      for i in range(0, self.size):
        check += (self.population[i].fitness / allfitness)
        #check1=1.0-self.population[i].fitness/allfitness
        #print(check1)
        if check >= randomnum:
          return self.population[i], i
        elif i == self.size - 1:
          return self.population[i], i
    if typeofsel=="rank":
      fitnessmass=[self.population[i].fitness for i in range(0,self.size)]
      decreaseindexes=argsort(fitnessmass)
      fitnessesprob = [0] * self.size
      for i in range(0, self.size):
        fitnessesprob[decreaseindexes[i]]=(2.0*(i+1.0))/(self.size*(self.size+1))
      randomnum = randint(0, 10000)
      randomnum = randomnum / 10000.0
      check = 0
      for i in range(0, self.size):
        check += fitnessesprob[i]
        if check >= randomnum:
          return self.population[i], i
        elif i == self.size - 1:
          return self.population[i], i
    if typeofsel in ("tournament_sel_2","tournament_sel_5","tournament_sel_9"):
      tour=0
      if typeofsel=="tournament_sel_2":
        tour=2
      if typeofsel=="tournament_sel_5":
        tour=5
      if typeofsel=="tournament_sel_9":
        tour=9

      tournir=[randint(0,self.size-1) for i in range(0, tour)]
      fitnessobjfromtour=[self.population[tournir[i]].fitness for i in range(0,tour)]
      return self.population[argmax(fitnessobjfromtour)], argmax(fitnessobjfromtour)


  def listpopulation(self):
    for i in range(0, self.size):
      self.population[i].display()

#############################################################
from math import cos,sin,exp,radians

import operator
OPERATORS = {'+': (operator.add,2), '-': (operator.sub,2),
             '*': (operator.mul,2), '/': (operator.truediv,2)}
    # ,
    #          'sin': (sin, 1),'cos': (cos,1),'exp': (exp,1)}

def add(ValuesList):
    sumtotal = 0
    for val in ValuesList:
      sumtotal = sumtotal + val
    return sumtotal

def sub(ValuesList):
    return ValuesList[0] - ValuesList[1]

def mul(ValuesList):
    return ValuesList[0] * ValuesList[1]

def div(ValuesList):
    if ValuesList[1] == 0:
        return 1
    return ValuesList[0] / ValuesList[1]


def Div(a,b):
  if b == 0:
    return 1
  return a / b

def sinx(a):
  xx = radians(a)
  if xx==float('inf'):
    return xx
  else:
    return sin(xx)

def cosx(a):
  with open('a.txt', 'w') as out:
    out.write('{}\n'.format(a))
  xx = radians(a)
  if xx == float('inf'):
    return xx
  else:
    return cos(xx)

def parab(a):
  return a**2

def minus(a):
  return -a

def sqrtx(a):
  if a<0:
    return float('inf')
  elif a==float('inf'):
    return a
  else:
    return sqrt(a)




addwrapper = funwrapper(operator.add, 2, "Add")
subwrapper = funwrapper(operator.sub, 2, "Sub")
mulwrapper = funwrapper(operator.mul, 2, "Mul")
divwrapper = funwrapper(Div, 2, "Div")
sinwrapper = funwrapper(sinx, 1, "Sin")
coswrapper = funwrapper(cosx, 1, "Cos")
expwrapper = funwrapper(exp, 1, "Exp")
parabwrapper = funwrapper(parab , 1, "x^2")
abswrapper = funwrapper(abs , 1, "Abs")
minuswrapper=funwrapper(minus,1,"Minus")
sqrtwrapper=funwrapper(sqrtx, 1,"Sqrt")


#def examplefun(x1, x2,x3,x4,x5):
def examplefun(x1,x2,x3):
  #return x1 * x1 + x1 + 2 * x2 + 1
  #return sinx(abs(x1))/abs(x1)
  #return 100*(x2-(x1**2))+((1-x2)**2)
  return x1*x1*x1+4*x2+8*x3+1
  #return parab(x)*sinx(x)+parab(y)*sinx(y)
  #return 10.0*sinx(pi*x1*x2)+20.0*(parab(x3-0.5))+10.0*x4+5.0*x5
  #return 0.5*(x1*x1+x2*x2)*(1.6+0.8*cosx(1.5*x1)*cosx(3.14*x2))
  #return 1-(0.5*cosx(1.5*(10*x1-0.3))*cosx(31.4*x1))+(0.5*cosx(22.36*x1)*cosx(35*x1))
  #return (0.1*x1*x1)+(0.1*x2*x2)-4*cosx(0.8*x1)-4*cosx(0.8*x2)+8
  #return (-10/(0.005*(x1*x1+x2*x2)-(cosx(x1)*cosx(x2/sqrt(2)))+2))+10
  #return (-100/((parab(x1)-x2)+parab(1-x1)+1))+100
  #return parab(x1)*abs(sinx(2*x1))+parab(x2)*abs(sinx(2*x2))-(1/(5*parab(x1)+5*parab(x2)+0.2))+5
  #return 0.5*(parab(x1)+x1*x2+parab(x2))*(1+0.5*cosx(1.5*x1)*cosx(3.2*x1*x2)*cosx(3.14*x2)*0.5*cosx(2.2*x1)*cosx(4.8*x1*x2)*cosx(3.5*x2))
  #return sqrtx(parab(x1)+parab(x2*x3-(1/(x3*x4))))
  #return parab(x1)+4*parab(x2)+9*parab(x3)+16*parab(x4)+25*parab(x5)+36*parab(x6)+49*parab(x7)+64*parab(x8)+81*parab(x9)+100*parab(x10)
  #return parab(x1) + 4 * parab(x2) + 9 * parab(x3) + 16 * parab(x4) + 25 * parab(x5)



#def examplefun(x):
  #return (sinx(abs(x)))/abs(x)

def constructcheckdata(count=300):
  checkdata = []

  for i in range(0, count):
    dic = {}
    #x = randint(0, 10)
    #y = randint(0, 10)
    x1=uniform(-5,5)
    x2 = uniform(-5, 5)
    x3 = uniform(-5, 5)
    #x4 = uniform(-2, 2)
    #x5 = uniform(-2, 2)
    #x6 = uniform(-5, 5)
    #x7 = uniform(-5, 5)
    #x8 = uniform(-5, 5)
    #x9 = uniform(-5, 5)
    #x10 = uniform(-5, 5)

    #x3 = uniform(0,1)
    #x4 = uniform(0,1)
    #x5 = uniform(0,1)
    #dic['x'] = x
    #dic['y'] = y
    dic['x1']=x1
    dic['x2'] = x2
    dic['x3'] = x3
    #dic['x4'] = x4
    #dic['x5'] = x5
    #dic['x6'] = x6
    #dic['x7'] = x7
    #dic['x8'] = x8
    #dic['x9'] = x9
    #dic['x10'] = x10


    #dic['x5'] = x5
    dic['result'] = examplefun(x1,x2,x3)

    checkdata.append(dic)
  return checkdata

if __name__ == "__main__":



  checkdata = constructcheckdata()
  checkdataforbesttree=constructcheckdata()
  print(checkdata)



  constantmas = [uniform(0,10) for i in range(1, 10)]

  launches=30
  res=[]
  percent=100
  averagenum_of_generation=0
  averagenum_len=0



  for i in range(0, launches):
    env = enviroment([mulwrapper,addwrapper,parabwrapper,subwrapper], ["x1","x2","x3"],
                     constantmas, checkdata, checkdataforbesttree,percent)


    returnedvalues=env.evolution()
    print("returnedvalues")
    print(returnedvalues)

    percent=returnedvalues[0]*100
    if returnedvalues[1]!=None:
      averagenum_of_generation+=returnedvalues[1]
      averagenum_len+=1
    if percent<=1:
      res.append(percent)
    #res.append(env.envolve()*100)
    print("launch number:")
    print (i)
    print("res")
    print (res)



  print("RES")
  print (res)
  print("reslen")
  print(len(res))
  averagenum_of_generation=averagenum_of_generation/averagenum_len
  print ("average num of generation")
  print (averagenum_of_generation)
