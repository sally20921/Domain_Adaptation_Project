# The @classmethod Decorator
- This decorator exists so you can create  class methods that are passed to the  actual class object  within  the function call, much like *self* is passed to  any other ordinary instance method in a class.
- In those instance methods, the *self* argument is the class instance object itself, which can then be used to  act on instance data.  *@classmethod* also have a mandatory first argument,  but this argument isn't a class instance, it's actually the uninstantiated class itself.

# **glob** - Unix style pathname pattern expansion
- The *glob* module finds all the pathnames matching a specified pattern according to the  rules used by the Unix shell, although results are returned in arbitrary order. 

# **ignite** - Events and Handlers
- To improve the *Engine* flexibility, an event system is introduced that facilitates interaction on each step of the run:
- engine is started/completed
- epoch is  started/completed
- batch iteration is started/completed

# The *@staticmethod* Decorator
- It  can be  called from an uninstantiated class object, although there is no *cls* parameter  passed to its method. 
- Since no *self* object is passed either, that means we also don't  have  access to any instance data,  and thus this method can not be called on an instantiated object either.

