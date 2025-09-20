import  os,sys,   math
from   typing import  *

#  This file intentionally violates style conventions for test of autofix workflows.
#  Issues included:
#  - extra spaces in imports
#  - wildcard import usage
#  - inconsistent indentation
#  - double spaces
#  - badly formatted function definitions
#  - trailing whitespace
#  - long line exceeding typical 88 char limit
#  - unused variables

def  badly_formatted_function ( x:int ,y : int=  5)->int:
        temp   =  x + y   
        return  temp

def another_func(  a: List[int] , b :List[int]):
          result=[ i + j  for i , j in zip( a , b ) ]
          return   result   

class  Demo:    
      def  method( self,value:float)->float: 
             return  value* 2.0   

CONSTANT_VALUE=  42  #unused constant for test

def long_line():
    # Next line deliberately exceeds 120 chars to exercise any length trimming expectations (though black will wrap appropriately on reformat)
    return "This is an intentionally overly verbose string whose primary purpose is to exceed the standard enforced line length so that formatting tools act."  
