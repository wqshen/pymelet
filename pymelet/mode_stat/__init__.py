# -*- coding: utf-8 -*-
# @Author: wqshen
# @Email: wqshen91@gmail.com
# @Date: 2020/6/10 15:25
# @Last Modified by: wqshen

docs = """
**Object-Based Diagnostic Evaluation (MODE)**

> developed at the Research Applications Laboratory, NCAR/Boulder, USA.
> Davis **et al.** (2006a, b) and Brown **et al.** (2007).


MODE was developed in response to a need for verication methods that can provide diagnostic information
that is more directly useful and meaningful than the information that can be obtained from traditional
verification approaches, especially in application to high-resolution NWP output. The MODE approach was
originally developed for application to spatial precipitation forecasts, but it can also be applied to other
elds with coherent spatial structures (e.g., clouds, convection).


MODE may be used in a generalized way to compare any two elds. For simplicity, eld1 may be thought of
in this chapter as "the forecast", while field2 may be thought of as "the observation", which is usually a gridded
analysis of some sort. The convention of field1/field2 is also used in Table 14.2. MODE resolves objects in
both the forecast and observed fields. These objects mimic what humans would call "regions of interest".
Object attributes are calculated and compared, and are used to associate ("merge") objects within a single
field, as well as to "match" objects between the forecast and observed fields. Finally, summary statistics
describing the objects and object pairs are produced. These statistics can be used to identify correlations
and dierences among the objects, leading to insights concerning forecast strengths and weaknesses.
"""
