

# taken from adcc (written by mfherbst)
def expand_test_templates(arguments, template_prefix="template_"):
    """
    Expand the test templates of the class cls using the arguments
    provided as a list of tuples to this function
    """
    parsed_args = []
    for args in arguments:
        if isinstance(args, tuple):
            parsed_args.append(args)
        else:
            parsed_args.append((args, ))

    def inner_decorator(cls):
        for fctn in dir(cls):
            if not fctn.startswith(template_prefix):
                continue
            basename = fctn[len(template_prefix):]
            for args in parsed_args:
                newname = "test_" + basename + "_"
                newname += "_".join(str(a) for a in args)

                # Call the actual function by capturing the
                # fctn and args arguments by-value using the
                # trick of supplying them as default arguments
                # (which are evaluated at definition-time)
                def caller(self, fctn=fctn, args=args):
                    return getattr(self, fctn)(*args)
                setattr(cls, newname, caller)
        return cls
    return inner_decorator
