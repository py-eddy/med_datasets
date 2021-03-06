CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ??j~??"?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?0?   max       P???      ?  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ????   max       >Z?      ?  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?Q???   max       @E?=p??
     ?  ?   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ????
=p    max       @vpz?G?     ?  '?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @+         max       @R            d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?3        max       @??          ?  /?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ?#?
   max       >?G?      ?  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?r   max       B-?6      ?  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A???   max       B-=t      ?  1?   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >?H}   max       C?&l      ?  2?   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?   max       C?'4      ?  3?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         @      ?  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      ?  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      ?  5?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?0?   max       P.?&      ?  6?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??0??(?   max       ???*0U3      ?  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ????   max       >Z?      ?  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??Q??   max       @E?=p??
     ?  8?   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??
=p??    max       @vl?\)     ?  @?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @          max       @R            d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?3        max       @?O@          ?  H?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A   max         A      ?  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ???X?e   max       ??bM???     P  J         )         k     ?            B      $      *      	      ,   "               *                     V   C   #         !   ;      5   2         N            N???O??@PF?,N7?~NPQPԐZN?qP?oN???N?)N@??P???O? ?P2??Nrj Pb??Oq?O;O.5YP3?DO?SM?9?N??O?J"N?O??qNUY5OrE4N.??Nuh?Nx??N?2dP??P?;?O??mM?0?O??BPsO?SUN??O?ګO???N??Nye?O?<?O$?-N??DN?I?NƷ??????9X??C??T???49X?o?ě???o??o?o%@  ;??
;ě?<t?<#?
<u<??
<?1<ě?<???<???<?`B<?<???=o=+=+=\)=t?=?w=?w=,1=49X=49X=8Q?=<j=D??=D??=Y?=]/=u=y?#=??=??=???=??T=??T>
=q>Z?qpt}???????????tqqqq???????????????????????0IV[\UI<6#
???>7?BNQSQNB>>>>>>>>>>????? ????????????????5Nl|?|[N@)??????????
????????!-B[g????????t[N5*VSUY[gt|????zthgf[VV88<HNTLH@<8888888888?????????????????????6[px????[B7)???? "/885341+"	&4CCNgt??????t[B)????????????????????????
/<LJE/???????86:<HUannvnkdaUHE<88??????	??????????????

???ggsy??????????????zg????????????????????)565-)%?? 

?????????f]gt??????????????tf???????? ?????????????????????????????????????????????????????
 -/0565/#	?o{???????{oooooooooo<@BNX[[[[[ZNEB<<<<<<a^gty????tlgaaaaaaaa????????????????????.79@Hmw}|{ytmaTH:3/.????)5BRG0????????!)BMPHINOKB;?????????????????????????????????????????? 
)5BNTTQB)
???????????????????????????????????????????63-'"??????????)6>:6*?????????

?????????sqtx?????????tssssss??????? 
'&
????VRQTXW[hjkrt|?}wth[V????????????????????../9<>CHLMKHD<;951/.//<AB<</##/////?n?z?~ÇÊÎÊÇ?z?n?m?n?o?n?j?i?n?n?n?nĚĦĿ??????????ĿĳĦĚčā?t?vĀāĉĚ?????̻лû??????????z?q?^?F?.?'?+?F?l??ŭŹ????????ŹŭŤţŭŭŭŭŭŭŭŭŭŭ?????&?????????????????????<?bŇŭŔœň?b?I?0?#???????????????ʼּ??????????ۼּʼ??????????ɼʼʼʼ??)?O?tĕĞěČā?t?6????????????????)????????????????????????????????????????D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??????????????????{??????????????????????"?;?C?G?9??ʾ??s?K?9?(?G?s???????ʾ??"?	??"?;?T?a?q?x?w?m?a?T?H?;?"??
????	?`?m?y???????????????m?T?G?;?:?4?4?<?G?`??????????????????????????????????????????(?A?M?I???:?<?(?????ڿӿܿÿ??ѿ??????????	?	?????????????????????????5?:?A?H?D?H?B?5?2?(????????(?+?5?4?A?M?Z?f?s?t?{???s?f?Z?M?E?A?6?*?*?4??????????????????????????????s?g?f?x???Z?^?f?s?v?p?h?i?i?f?Z?M?A?4?.?*?/?4?A?Z?Z?]?`?`?Z?Z?W?M?K?M?Q?V?Z?Z?Z?Z?Z?Z?Z?Z?r?v?~?????~?{?r?e?]?e?q?r?r?r?r?r?r?r?r?(?4?M?Z?a?r?w?s?f?Y?M?A?4?(?????#?(?????ʾ׾??????????׾ʾ?????????????????ù?????????#?,??????????øòïæìù?A?N?S?Y?Z?\?_?Z?Y?N?H?A?>?;?A?A?A?A?A?A?`?m?y???????????y?r?m?`?T?G???:?;?=?G?`????????????????????????????????????T?a?e?l?a?U?T?H?;?;?;???H?L?T?T?T?T?T?T?"?/?4?3?/?(?"??????"?"?"?"?"?"?"?"?<?H?T?M?H???<?/?#?+?/?5?<?<?<?<?<?<?<?<???
?#?<?=?,?#???????????ĸĴĴĿ??????¿????????½?t?[?B? ??)?5?B?[¦¿??!?2?=?F?O?S?F?:?!?????????????????
??????
???	?
?
?
?
?
?
?
?
?
?
????*?2?=?E?K?C?6?*???????????????hƁƚ????????????????Ƣ?y?h?]?\?R?h?ùϹ???????'?'??????ܹù??????????ýl?y?~?y?s?l?b?`?_?Y?`?d?l?l?l?l?l?l?l?l???????ûֻ޻߻ٻлû????????????????????Y?r?~???????????????r?d?Y?D?@?9?6?7?@?YDIDVDbDoDxDsDoDbDVDIDFDEDIDIDIDIDIDIDIDI????????????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?EwEwE~E?E?E???'?4?@?M?Y?f?k?f?Y?M?K?@?4?'??????zÇÍÓààààßÓÇÅ?z?y?v?u?z?z?z?z???????????????????ŹŭŨŭŹ????????ǭǭǨǡǔǈǇǃǃǈǈǔǡǦǫǭǭǭǭǭ Z * + @ N ) .   0 r ] W = ) G > 9 D . E ? 7 / ? E p C ~ O Y i : k E j * h S h : / < V B I [ o =    ?    <  =  h  ?  ?  ?    +  ?  V  ?  ,  z  ?  K  ?  ?  ?  %  ?  /  x  ?    ?  ?  ?  ?  ?  ?  ^  ?  ?  <    ?  u  Q  {  7  ?  ?  ?  r  ?  T  ޼#?
;o<ě??t???`B=???;?`B>?G?<#?
;??
;??
=?hs<?`B=49X<?1=e`B=?P<?=?P=?O?=m?h<???=t?=<j=?w=???=??=u='??=8Q?=49X=H?9>?=?S?=???=L??=???=??T=?S?=ix?=?l?=?S?=?E?=??->?-=???=?^5>?w>uB
?yBW4B$?B?B<\B?zB??B	9?B	BB B?BkkA?rB??B??B??B?UB?B#?:B?B?8Bj?B$H|B
?#B]TBB?`B9GB)?B>?B	?{B??A???B&?B(`B?WB
?B?B??B-?6B?zB?UB?BѽB??B?B!O/B?IB??B
??BPB$9B?BGB??B7TB	??B	??B'?B4lB>?A???B	A?B??B%?BD?B??B#?*B@?B<?B??B$BhB
?ZBJ?BI?B?!B?B)8B@B	?5BD*A???B?}B??B?B?XB?>B~IB-=tB??B?B?#B?qB?xB??B!?]B??B,?A?\]A?z?@?@UA?]A??A??jA `)A?<A???C?;+A???AO61A??Ai?FA??\A?ɡA?_A???A>LA???A=?A?k?????A;??AQҌA?4xA?jCAjp@U?;A???A?}?A?y?A?'A?U@g??A??>A?9YBb>?H}A?0@?G????{C?|?@?)?C?&l@???Aə?A???BQYAȀ{A??[@?7A?~?A??A??@?:0A?A??C?8A?y?ARW7A?wVAl??AЋ?A??[AҀsA??zA= A?p?A<??A?$???[A;?AQ??A?}?A???Ai??@S?_A??eA???A?h?A???A??W@k??A??bA???B?>?A>Y@??=@??C?y?@???C?'4@?;`Aɇ?A??JBG?         *         l     @            C      $      *      
      -   "               +                     V   D   #         "   ;      6   3         O                     3         G      9            O      -      1            +                  '                     %   ?   %         -            #         !                     '                           +      %      !            !                  #                        /   %         +            #                     N???O?O??RN7?~NPQO?3N?qO??N??6N?)N@??P
nOZ?aP ?NLDO?%?N? O;O.5YO?9?OvׯM?9?N??O?J"N?O?,?N?OOrE4N.??Nuh?N;cN?2dOG?0P.?&O??mM?0?O??BP??OlcN??O??OڞqN*@SN0?OR??O$?-N??DN?I?NƷ  d  -  ?  ?  ?  ?  ?  ?  ?  ?  	  ?  ?  ?  ?  ?  u  F  ?  ?  ?  >  ?  ?  ?  ?  ?    O    ?  ~  ?  ?  ?  ?  ]  p  
z  =  ?  [  ?  f    ?  w  ?  
'?????u?ě??T???49X=u?ě?>)??%@  ?o%@  =?P<D??<D??<49X<??h<???<?1<ě?=?P<???<?`B<?<???=o=\)=C?=\)=t?=?w='??=,1=??
=]/=8Q?=<j=D??=H?9=q??=]/=?o=?+=?\)=?7L=ȴ9=??T=??T>
=q>Z?qpt}???????????tqqqq????????????????????????
#0BDA:/+
?????>7?BNQSQNB>>>>>>>>>>????? ???????????????#)-..-'?????????
????????@?AFN[gt??????tg[NE@VX[dgktvyttg^[VVVVVV88<HNTLH@<8888888888??????????????????6BO_giovvh[QB5"/3//.+%"	! "5GL[gt??????t[N5!???????????????????????????1<<9/#?????:8<FHUUakbaUH<::::::??????	??????????????

???sqw~??????????????zs????????????????????)565-)%?? 

?????????f]gt??????????????tf???????? ?????????????????????????????????????????????????????
 -/0565/#	?o{???????{oooooooooo<@BNX[[[[[ZNEB<<<<<<fdgtt????ttgffffffff????????????????????F?=>CHHTaimpqnmfaTHF????)5HLB5)????!)BMPHINOKB;??????????????????????????????????????????
)5BNSSPB5)???????????????????????????????????????????+/+$??????????)69:6)#
????????


?????????rt|???????utrrrrrrrr???????

????VRQTXW[hjkrt|?}wth[V????????????????????../9<>CHLMKHD<;951/.//<AB<</##/////?n?z?~ÇÊÎÊÇ?z?n?m?n?o?n?j?i?n?n?n?nčĚĦĳļĿ????ĿļĳĦĥĚčĄĄĊčč?????????????????????x?_?F?6?:?E?V?_?l??ŭŹ????????ŹŭŤţŭŭŭŭŭŭŭŭŭŭ?????&?????????????????????#?0?<?I?N?V?R?I???0??
???????????
??ʼּ??????????ۼּʼ??????????ɼʼʼʼ??B?O?[?h?s?y?z?u?l?[?O?B?6?$???"?+?6?B????????????????????????????????????????D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D?D??????????????????{????????????????????????ʾ׾????????׾???????i?n????????"?/?;?H?T?a?l?r?m?a?H?;?/?"??????"?G?T?`?v???????????????`?T?G?A?=?7?5?:?G????????????????????????????????????????????(?4?@???2?(??????????????????????????????????????????????????????5?:?A?H?D?H?B?5?2?(????????(?+?5?4?A?M?Z?f?s?t?{???s?f?Z?M?E?A?6?*?*?4????????????????????????????????~???????M?Z?f?s?u?p?h?i?h?f?Z?M?A?4?.?+?4?6?A?M?Z?]?`?`?Z?Z?W?M?K?M?Q?V?Z?Z?Z?Z?Z?Z?Z?Z?r?v?~?????~?{?r?e?]?e?q?r?r?r?r?r?r?r?r?(?4?M?Z?a?r?w?s?f?Y?M?A?4?(?????#?(?????ʾ׾??????????׾ʾ?????????????????ìù????????!?*???????????úóðèì?A?N?Q?W?Z?Z?Z?N?I?A???>?A?A?A?A?A?A?A?A?`?m?y???????????y?r?m?`?T?G???:?;?=?G?`????????????????????????????????????T?a?e?l?a?U?T?H?;?;?;???H?L?T?T?T?T?T?T?"?/?1?/?/?#?"????? ?"?"?"?"?"?"?"?"?<?H?T?M?H???<?/?#?+?/?5?<?<?<?<?<?<?<?<???????????
?
???
???????????????????¿???????????????t?]?Q?O?F?B?N¦¿??!?2?=?F?O?S?F?:?!?????????????????
??????
???	?
?
?
?
?
?
?
?
?
?
????*?2?=?E?K?C?6?*???????????????hƁƚ????????????????ƣƍ?z?j?^?]?h?ùϹ??????????????ܹù??????????ýl?y?~?y?s?l?b?`?_?Y?`?d?l?l?l?l?l?l?l?l???????ûԻܻݻֻлû????????????????????Y?r?~???????????????~?j?]?Y?L?@?9?@?L?YDIDVDbDbDoDoDoDbDVDIDHDHDIDIDIDIDIDIDIDI????????????????????????????????????????E?E?E?E?E?E?E?E?E?E?E?E?E?E?E~E|E?E?E?E???'?4?@?M?Y?f?k?f?Y?M?K?@?4?'??????zÇÍÓààààßÓÇÅ?z?y?v?u?z?z?z?z???????????????????ŹŭŨŭŹ????????ǭǭǨǡǔǈǇǃǃǈǈǔǡǦǫǭǭǭǭǭ Z + , @ N C .   0 r ^ X 6 * E Y 9 D $ A ? 7 / ? C a C ~ O f i ' \ E j * d F h 5 2 D G 9 I [ o =    ?  S  %  =  h  R  ?  `  ?  +  ?  ?  ?  ?  g    ?  ?  ?  ?    ?  /  x  ?    [  ?  ?  ?  =  ?  ?  ?  ?  <    G    Q  E  ?  W  W  ?  r  ?  T  ?  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  d  c  Y  N  A  '  ?  ?  ?  ]  .    ?  ?  ?  ?  e  B    ?  ?      $  +  ,  ,  $    ?  ?  ?  ?  W  !  ?  ?  <  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  W  1  %      ?  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  o  f  \  k  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  m  `  T  I  >  3  '    ?  ?    %  !      P  m  ?  ?  ?  ?  ?    b  ?     ?  ?  ?  ?  y  j  V  ?  "     ?  ?  ?  q  K     ?  ?  _  (  !  ?  ?  ?  Z  ?  ?  ?  a  ?  G  ~  Y  ?  @  :  ?  v  }  
?  "  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  X  >     ?  ?  ?  Q  ?  ?  ?  k  R  9       ?  ?  ?  ?  ~  `  ?    ?  ?  g    	      ?  ?               "  #                ?    F  k    ?  ?  ?  ?  ?  ?  [  #  >    ?  ?  	  a  +  v  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  5  ?  ?  |  F  	  ?  g    ?  ?  ?  ?  ?  ?  ?  H      $  7  D  ;    g  6  ?    ?  }  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  ]  A  #    ?  ?  ?  u  ?    ?  ?  D  q  ?  ?  ?  t  [  C  ,    ?  ?  Y  ?  9  "  U  Z  d  j  o  s  p  b  L  2    ?  ?  q  5  ?  ?  ?  V  !  F  <  2  )  !  #  (  %        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  g  O  2    ?  ?  ?  B   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  \  )  ?  ?  i  ?  ;   ?  ?  ?  ?  ?  ?  ?  ?  ?  |  w  b  =    ?  ?  Q  ?  $  ?  ?  >  7  0  *  #          ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  l  a  T  C  *    ?  ?  ?  Q  ?  ?  ?  ?  r  d  U  F  7  &      ?  ?  ?  ?  ?  ?  g  K  ?  ?  ?  ?  ?  }  \  8    ?  ?  w  2  ?  ?  p  .  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  e  N  4       ?  ?  ?  \    ?  ?  ?  ?    X  ,  ?  ?  ?  M    ?  w  +  ?  ?    ?  O  C  7  +        ?  ?  ?  ?  ?  ?  ?  {  g  I  (     ?          ?  ?  ?  ?  ?  ?  ?  ?  ?  |  t  l  V  ?  ?  <  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  a  K  6  #    ?  ?  ~  g  Q  :  #            ?  ?  ?  ?  ?  j  4  ?  ?  ?  	Y  	?  
m  
?    f  ?  ?  ?  ?  ?  ?  >  
?  
U  	?  ?  m  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  (  ?  1    ?  c  ?  ?  ?  ?  ?  _  /  ?  ?  p    ?  ?  v    ?  A  ?  ?  T  ?  ?  ?  z  j  ]  T  L  C  :  0  #    
  ?  ?  ?  ?  ?  ?  ]  V  I  9  %    ?  ?  ?  o  7  ?  ?  o    ?  \  ?  ?  c  m  d  K  2    ?  ?  ?  ?  ?  t  K  !  ?  ?  U  ?  r  ?   ?  
7  
U  
z  
v  
e  
O  
0  
  	?  	?  	F  ?  ?  5  ?  `  ?  r  ?  T  =  3  )      
  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  p  _  M  ?  ?  ?  ?  ?  ?  Z  %  ?  ?  M     ?  R  ?    )  A  ?  <  8  S  Z  T  C  )    ?  ?  E  ?  ?  K    ?  T  ?  ?  5  b  ?  ?  ?  ?  ?  ?  ?  ?  X    ?  v  '  ?  ?  2  ?    ,  ?  L  W  `  e  a  Y  I  *    ?  ?  ?  U  %  ?  ?  ?  N    ?  ?  *  ?  ?  ?        ?  ?  k  ?  s  
?  	?  ?  ?  ?  2  ?  ?  ?  ?  z  \  |  y  o  ^  H  1    ?  ?  ?  [  .  
  ?  ?  w  _  G  (  	  ?  ?  ?  o  ;  ?  ?  }  A    ?  ?  Z      ?  ?  ?  ?  |  b  C    ?  ?  ?  t  A  	  ?  ?  n  F    -  ?  
'  	?  	?  	i  	+  ?  ?  i  #  ?  ?  0  ?  ?  '  ?  i    ?  $