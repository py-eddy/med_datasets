CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ???$?/      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M??^   max       P???      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ????   max       =?Q?      ?  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?E?Q??   max       @E?p??
>     ?   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???Q??     max       @vvz?G?     ?  (?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @P?           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @???          ?  1?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??j   max       >I?      ?  2?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A??6   max       B-!7      ?  3?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A??J   max       B->      ?  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >?G?   max       C?m?      ?  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?}B   max       C?gc      ?  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          T      ?  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      ?  7?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      ?  8?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M??^   max       P?5      ?  9?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??m??8?Z   max       ??Ϫ͞??      ?  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ????   max       =?Q?      ?  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?E?Q??   max       @E?p??
>     ?  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??z?G?    max       @vvz?G?     ?  D?   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @P?           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @?w?          ?  M?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @?   max         @?      ?  N?   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??*?0??   max       ???t?j     0  O?         	   M   
   E      $   '   
   6   +   
   ;            T         *         +   !                  )                  #         2         "      	   %   9               8      6   N?g?N[܎N?PvO???Nz?'Pa,XN??O??fO???O??Pm?O?*#N!??PX?N$??OܞN???P???O2?N??iP!?N??O?HO?CO?.?O???N0??N/?OHA?O 4P?N?Q?N??|N??~O??aO\8OpB?M??^O6s?O??.O=??NȔN??[NX?CN?_?O?8P?.N?
"N??BOmN3AROtFN???O???O=1???????ͼ????D???D??;o;o;??
;??
;??
;?`B;?`B<o<t?<49X<49X<e`B<u<u<?C?<?t?<?j<?j<ě?<???<???<?`B<??h<??h<?<?<?<???<???=t?=?P=?P=?P=??=?w='??=0 ?=49X=49X=8Q?=ix?=}??=}??=?%=??=?C?=?hs=???=???=?Q?-)(0<INUYUJI<0------???????????????????????? 
??./9@N[g??zxvwtg[N5.prt|????????ytpppppp???#<HUaltunU#???????????????????????????????
/90.#
???????
/9BGZ]ZUH/?????????
#/234/#
???????????  ?????????????????????????????????????????????eaht??????????????re).-)%|y~???????????????||?????????????????????????58/.31)?????)6BENUPPHB-&[a][ZY[\hstxxvtrih[[????'-9,#
???????????6BOSY^^ZOB6\Y[`alz?????????zne\????????????????????

#-08<BA><0#
????????????????????	()))?????????????????????????????????????????????? 8IUURB)??????????????????????????????????????????
#&&##
 ??))./7465+ :777;=HTanz}zma]TH;:torvwz????????????ztffhmtxyztsphffffffff?????????????? #/<H\a_YPSOH</) ??????

???????><;BGNNONB>>>>>>>>>>../4<HNUTPH<1/......?????????????????????????????????????????????$ ?????????	??????	),,)  wttqqz}?????????zww????????????????????|{????????||||||||||????????????????????????????????????????)17;;6)????????

???????˻????????????????????????????????????????????????????????? ???????ûлܻ???????????????ܻлȻû?????????6?O?[?n?t?s?h?[?I?6?)???????????????????????????????????????????????????޾????о??????ܾʾ????s?d?b?R?Z?f?????????'?3?>?@?A?@?<?5?3?)?'????'?'?'?'?'?'???????5?[?b?]?B?'????????????????޿y?????????????y?m?G?;?.?(?$?,?A?]?`?l?y???(?5?8?5?2?-?(?!??????????	??ìù????????????? ????????÷ôìÞàì???(?4?A?G?J?G?A?4?????????ݽܽ???????? ???? ?????????????s???????????????s?W?5?(??????(?1?Z?s?????????????????????????????????????????#?/?<???H?O?Q?H?D?<?9?/?&?#???? ?#?#?;?G?L?M?P?G?;?.?-?&?.?8?;?;?;?;?;?;?;?;?????????????տſ????y?`?T?I?D?I?]?g?????????????????????y?m?`?T?T?X?`?m?o?y?????[?f?s?x?????????????????s?i?f?Z?Y?Y?[?	??? ???????????????????????????????	?"?.?;???;?.?(?"?????"?"?"?"?"?"?"?"?׾????	????
???????׾ʾ??????????ʾ?E?E?E?E?E?E?FFFE?E?E?E?E?E?E?E?E?E?E;M?Z?_?f?y????s?b?M?A?(??????(?A?M???????????????????????r?p?e?_?j?r?v????
???????
??????????????????????ÓàâàÛÓÉÇÇÂÇÊÓÓÓÓÓÓÓÓ???!?-?:?D?F?Q?R?F?A?:?-?!??????????????*?/?6?:?B?6?*???
??????????O?hƃƚƧ????????????ƧƚƎ?}?s?q?h?T?O?T?`?m?y?|??|?y?m?`?\?T?L?H?T?T?T?T?T?T?????????????????????????????????????????ּ?????????????????׼ּϼͼּּּּּ??#?0?<?I?O?U?b?d?f?b?U?<?0?#?
??????
?#?/?;?H?T?^?a?f?i?i?m?g?a?_?X?O?H?9?+?.?/???)?B?O?S?\?`?[?U?O?B?6?)???	???????????????????????????????????????????āčĚĦıĽĿ??ĿľĳĦĜĚčā?y?u?{ā????????????????????ùìåáìù???n?z?|?z?r?s?w?p?a?[?H?<?6?+?,?/?<?H?U?n?)?5?B?L?B?6?5?)?'?"?)?)?)?)?)?)?)?)?)?)EEE!E*E2E/E*EEED?D?EE
EEEEEE?zÄ??{?z?n?h?a?U?Q?U?[?a?l?n?x?z?z?z?z?Ϲ׹ܹ????޹ܹϹùùù??????????ùǹϹ???????????????????????????????y?z????????"????'?(?????ɺ????????????ɺ???r?~?????????????????~?r?o?p?r?r?r?r?r?r?_?l?x?????????????x?u?l?_?S?H?N?S?Y?_?_?????????	???????	???????????????׼'?4?@?@?H?@?4?'?#??'?'?'?'?'?'?'?'?'?'???????ּ?????????????ּʼü???????????????????
???
?????????????????????????@?M?Y?\?b?d?]?Y?L?@?4?'????	???&?@E?E?E?E?E?E?E?E?E?E?EuEsEqEiEiEoEkEuE?E? ^ d U : , * P F ^ d E $ > V < 7 9 ( o [ E ?   ` & V Y = C O * P - C k + ? D 8 F d  ? 5 0 Z > c . ? H + L O    ?  ?    "  |  ?  ?  o  ?  z  ?  ?  7  ?  G  H  ?  >  ?  ?  ?  (  y  F  ?  ?  v  ;  ?  &  ?  ?  )  ?  1  ?  ?  +  ?  ?  ?  &  ?  ?  ?  ?  ?  ?    k  J  '    !  `??9X??j?#?
=??P;ě?=?hs<D??='??=49X<e`B=u=L??<?t?=?C?<u=o<?t?=ȴ9<?9X<?j=q??<???=L??=?7L=m?h=8Q?=o=+=aG?=8Q?=?\)=??=??=?P=D??=D??=?hs='??=?%=?E?=y?#=<j=??w=]/=Y?=??=??F=?C?=??P=??-=??
>%=?Q?>I?=?x?B&Y BO?B$Y_B??B
P?B>B!5?Bg?BĦB#`B,?B xOB!?oB9?B?7B??B?>B??BpWB?PBh?B?B??B?2Bg?B%:<B?B??B g?B?OB?SB-!7B??B$??BZ?A??6B ?6BoB=B?uB*FB?kB?LBN?B?+B?B?B??B?B|?B?BL?BPhBx?B??B& ?B??B$LB??B
J9B?DB!5?B?3B?B?zB[#B ??B!??B' B??B?BB??B?)B=?B??B?PB??B??BD?B@B%8?B}B??B ??B?zB??B->B)B$E?B?.A??JB ?B??B=?B?}BC?B?;B? B??B??B@B>pB?_B4[B?B=?BeOBKB@B=b@?QjA??O@???A???AЦ?AJ????uA???Ah??A?p?Aϡ?A3~K@?5A?g?@???A´Ac??Aq?Amh?ADo?A?H?A`MCAT?|C?m?A:?[@??
A?f{A??@n?PA?o?BɱAi??Ar?\A?1A??A???A?D+@?{A߼`A?٬AŇAA??"C?mWAǍ?>?G?A??@C??@	[?@??A?n]@ͳjA 71A???@?n>C???@??A??@???A?c?AЃAJ???w,A?w?Aj?[A?u?A?U?A2??@?A?p?@?7?AAcJ?Arg?An??AF??A?X?Aao?AT??C?gcA:??@?zdA?}A?U\@sԎA?|EB͊Aj?xAt??A??A?uqA???A?S?@??A߉uA??A?{(A?~C?hAƕ?>?}BA???@L?@?e@??SA?@?	bA ?#A??p@д?C??         
   N   
   F      %   (   
   6   ,      <            T         *         ,   "                  )   	               #         3         #      	   &   :               8      7               #      3      '   '      '         3            7         +            !                  -                                             !   1                                          !      #         !                                                                                                            +                        N?g?N[܎N?PvOR ?NN??O???NH?7O?K(O?Y?O??O?8~OVΣN!??O?6N$??N???N???O?g?Nw?1N??iOq??N??OfэO?CN?ȽO"r,N0??N/?N???O 4OH?N?Q?N??|N??~O??aO\8O!?[M??^O6s?N??\O+??NȔN??[NR?N?E?O?y?P?5N?
"N??BO?N3ARO??N???OS?&O=1  ?  ?     
?  ?  Y  m  M  ?  ?  ?  3  ?  V  ?  ?  ?  p  ?  @  A  ?  ?  G  ?  ?    ?  ?  ?  ?  \  ?  ?  ?  ?  y  d    B  ?  ?  
  
6  [  e  ?  ?  ?  ?  3  
?    
?  	???????ͼ???<??h?o<??h;??
<e`B<?o;??
<u<??
<o=t?<49X<?C?<e`B=ix?<?t?<?C?=t?<?j<?<ě?=?w<?`B<?`B<??h=?P<?=@?<?<???<???=t?=?P=8Q?=?P=??=?o=,1=0 ?=49X=<j=<j=q??=?o=}??=?%=?+=?C?=?{=???=?E?=?Q?-)(0<INUYUJI<0------???????????????????????? 
??B@BGN[gsvuqnkge[SNHBstt???????ztssssssss?
#/<PX^\YUE</#???????????????????????????????
$('
????????
#/<BHRSH</*??????
#/234/#
???????????????????????????????????????????????????????{wvz??????????????{).-)%???????????????????????????????????????????????????)68BGB>60)[a][ZY[\hstxxvtrih[[??????
 " 
???????)6BDORWWPB6)#\Y[`alz?????????zne\????????????????????
#*06<@@<0#
????????????????????	()))????????????????????????????????????????)8BEIHBB5)?????????????????????????????????????????
#&&##
 ??))./7465+ :777;=HTanz}zma]TH;:yxyz}????????????zzyffhmtxyztsphffffffff??????????????)&)./<HJROHH</))))))????

????????><;BGNNONB>>>>>>>>>>../4<HNUTPH<1/......???????????????????????????????????????????????#!??????????
??????	),,)  wttqqz}?????????zww????????????????????|{????????||||||||||???????????????????????????????????????&)/485)?????????

???????˻????????????????????????????????????????????????????????? ???????ûлܻ???????????????ܻлȻû????????)?6?B?O?Z?d?d?[?O?B?6?)????????)???????????????????????????????????????޾????žξҾԾҾ?????????z?s?o?r?z???????'?3?;?<?8?3?'?"?? ?'?'?'?'?'?'?'?'?'?'??????)?5?A?N?W?T?B?)???????????????m?y?????????????y?m?`?G?;?/?1?6?;?J?`?m???(?5?8?5?2?-?(?!??????????	??ù????????? ?????????????þøøñåëù????(?4?8?@?<?4?(???????????????????? ???? ?????????????g?s???????????????????g?Z?N?H?D?F?N?Z?g?????????????????????????????????????????#?/?<?H?I?I?H?@?<?3?/?,?#?#???#?#?#?#?;?G?L?M?P?G?;?.?-?&?.?8?;?;?;?;?;?;?;?;?????ÿȿƿ????????????p?m?f?f?m?y???????y????????????y?p?m?f?m?t?w?y?y?y?y?y?y?[?f?s?x?????????????????s?i?f?Z?Y?Y?[?????????????????????????????????????????"?.?;???;?.?(?"?????"?"?"?"?"?"?"?"?ʾ׾????????
?	???????׾ʾ????????¾?E?E?E?E?E?E?FFFE?E?E?E?E?E?E?E?E?E?E;4?A?M?T?Z?[?[?Z?M?A?4?(?%??(?2?4?4?4?4???????????????????????r?h?b?f?l?r?y????
???????
??????????????????????ÓàâàÛÓÉÇÇÂÇÊÓÓÓÓÓÓÓÓ???!?-?:?F?H?J?F?:?.?-?!? ??
?????????*?/?6?:?B?6?*???
?????????ƚƧƳ??????????ƸƳƧƚƎƍƆƄƉƎƕƚ?T?`?m?y?|??|?y?m?`?\?T?L?H?T?T?T?T?T?T?????????????????????????????????????????ּ?????????????????׼ּϼͼּּּּּ??#?0?<?I?O?U?b?d?f?b?U?<?0?#?
??????
?#?/?;?H?T?^?a?f?i?i?m?g?a?_?X?O?H?9?+?.?/?)?6?B?F?O?V?Z?O?B?6?)?????????)????????????????????????????????????????āčĚĦıĽĿ??ĿľĳĦĜĚčā?y?u?{ā????????????????????????????????????n?y?q?r?u?n?a?]?H?<?8?/?,?-?/?<?H?U?\?n?)?5?B?L?B?6?5?)?'?"?)?)?)?)?)?)?)?)?)?)EEE!E*E2E/E*EEED?D?EE
EEEEEE?zÁ?~?z?z?n?a?\?a?m?n?x?z?z?z?z?z?z?z?z?Ϲչܹ޹??ܹڹϹù¹????????ùɹϹϹϹ????????????????????????????????{?{??????????????&?'?????ɺ??????????ɺ????r?~?????????????????~?r?o?p?r?r?r?r?r?r?_?l?x?????????????x?u?l?_?S?H?N?S?Y?_?_???????	???????	???????????????????'?4?@?@?H?@?4?'?#??'?'?'?'?'?'?'?'?'?'?????ʼּ????????????????ּʼ???????????????????
???
?????????????????????????@?M?Y?_?`?X?M?H?@?4?'??????'?(?4?@E?E?E?E?E?E?E?E?E?E?EuEsEqEiEiEoEkEuE?E? ^ d U 7 5 0 M N Z d B ' > - < F 9  a [  ?   . $ V Y N C  * P - C k / ? D ' A d  ? " ! R > c ' ? 2 + K O    ?  ?    ?  U  ?  ?  ?  ?  z    ?  7     G  ?  ?  ^  ?  ?  ?  (  ?  F  ?  S  v  ;    &  ?  ?  )  ?  1  ?  _  +  ?  ?  ?  &  ?  |  ?  u  ?  ?    5  J  S    ?  `  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  @?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  g  [  J  7  %       ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?         ?  ?  ?  ?  ?  ?  ?  y  n  o  m  f  [  D  !  ?  &  ?  ?  	6  	?  
  
e  
?  
?  
?  
?  
i  
  	?  	  h  ?  ?  ?    &  ?  ?  ?  ?  ?  ?  ?  v  j  ]  N  @  1  !    ?  ?  ?  ?  y  %  ?  ?    1  C  P  W  X  Q  F  +    ?  ?  !  ?  ?    ?  ^  c  g  i  k  l  l  k  j  g  c  _  [  X  U  S  U  W  Y  [  ?    /  >  H  K  C  8  #    ?  ?  ?  S    ?  J  ?  ?    ?    D  b  z  ?  ?  o  Q  !  ?  ?  ?  ?  [  ?  ?  ?  ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  l  T  <  2  ;  :  &    ?  ?  ?  v  y  ?  ?  ?  ?  ?  ?  V  /  /  ?  ?  r  >  ?  ?  >  ?  7  ?  ~  ?  ?    *  2  2  -  !    ?  ?  ?  W  ?  ?  9  ?  ?     ?  ?  n  R  6  "        	  ?  ?  ?  ?  ?  ?  q  U  7    H  ?    g  ?  ?  3  K  U  V  L  6    ?  ?  U  ?  F  w  (  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  n  a  U  I  .    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  b  6    ?  ?  p  >  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  h  [  Z  ?  ?    U  ?  ?  $  S  l  n  Y  9    ?  -  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {    ?  z  m  `  @  ;  6  1  )  !        ?  ?  ?  ?  ?  ?  ~  d  D  #      8  P  `  x  ?    /  >  ?  1    ?  ?  ?  -  ?    N  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  u  o  i  c  ]  W  Q  |  ?  ?  ?  ?  ?  ?  ?  ?  ?  n  L  &  ?  ?  ?  Z  	  ?   ?  G  "  ?  ?  ?  Q    ?  ?  ?  o  G    ?  ?  %  ?  ?    U  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  _    ?  P  ?  ?  ?   v  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  g  E  $  ?  ?  ?  k  T  0  ?      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    p  Z  A  )    ?  ?  ?  ?  ?  ?  ?  ?  ~  ]  <  "    ?  ?  ?  ?  ?  ?  i  L  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  Z  -  ?  ?  9    ?  ?  e  ?  s  ^  J  6  !      ?  ?  ?  ?  t  N  &    ?  ?  ?  ?  ;  E  C  P  n  s  w  }  ?  }  k  W  :    ?  ?  5  ?      \  W  R  L  F  ?  5  *      ?  ?  ?  ?  ?  e  ;     ?   ?  ?  ?  ?  ?  ?  ?  ?  q  \  F  0      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  c  R  <  &    ?  ?  ?  ?  N     ?   ?   ?   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  L  &  ?  ?  U  ?  ?  ?  ?  ?  p  N  /  <  <  '    ?  ?  x  0  ?  ?  ?  ?  ?  )  M  h  v  w  h  O  1    ?  ?  a    ?  q  "  ?  o  ?  d  Y  N  C  8  )    ?  ?  ?  ?  ?  ?  ?  ?  ?    @  z  ?      ?  ?  ?  ?  r  I    ?  ?  ?  G    ?  u  )  ?  ?  )  U  ?  ?  ?  ?  ?  ?  ?    <  <  1    ?  ?  %  ?  ?  6  ?  ?  ?  ?  ?  ?  ?  ?  s  [  =    ?  ?  ?  ?  q  F  I  U  `  ?  ?  ?  ?  ?  ?  ?  ?  z  h  U  C  0      ?  ?  ?  ?  ?  
  	?  	?  	?  	b  	1  ?  ?  ?  O    ?  w  $  ?  y    ?  ?  U  ?  	  	=  	?  
:  
N  
a  
u  
?  
?  
?  
?  l    ?    ?  ?  ?  ?  6  E  U  W  P  H  ;  -       ?  ?  ?  [  +  ?  ?  ?  ^  "  #  c  [  I  '    ?  ?  ?  ?  ?  ?  \    ?  L  ?  ?  ?  s  ?  ?  ?  ?  ]  .    ?  ?  ]  %  ?  ?  O  ?  ,  ?  ?  ?  ?  ?  ?  ?  ?  ?  x  l  [  H  5       ?  ?  ?  ?  ?  ?  ?  ?  ?  z  b  F  (  	  ?  ?  ?  ?  ?  ?  ?  j  8     ?  ?      ?  ?  ?  ?  ?  ?  ?  ?  m  V  7    ?  ?  m  -  ?  ?  z  Q  3  3  /  (    ?  ?  ?  ?  ?  q  W  ?  (      *  ?  W  r  	?  
?  
?  
?  
?  
?  
?  
?  
?  
?  
D  	?  	?  	  ?  ?  g  ?  ?  ?      ?  ?  ?  ?  ?  ?  o  U  @  %  ?  ?  ?  `  "  {  ?   ?  
?  
?  
?  
?  
?  
?  
z  
L  
  	?  	y  	  ?  +  ?  %  ?    W  s  	  ?  ?  ?  =  ?  ?  k  G  H  `  r  l  5  ?  ?  ?  ?  ?  u