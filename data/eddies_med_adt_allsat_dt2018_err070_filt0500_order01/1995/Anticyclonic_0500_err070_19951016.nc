CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?`bM????   max       ???\(?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N?   max       P?.      ?  ?   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ?H?9   max       =???      ?  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>??\)   max       @E??Q??     ?   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?????R    max       @vc?z?H     ?  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @+         max       @P@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?r        max       @?i?          ?  2?   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??w   max       >??w      ?  3?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?w?   max       B-?      ?  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?ow   max       B-=?      ?  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >IG?   max       C???      ?  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Ah$   max       C???      ?  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         
      ?  7?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      ?  8?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      ?  9?   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N?   max       P?}?      ?  :?   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??J???D?   max       ??C?\??O      ?  ;?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ?D??   max       >5?}      ?  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @>?=p??
   max       @E?=p??
     ?  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ?׮z?H    max       @vc?z?H     ?  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @+         max       @P@           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?r        max       @?̀          ?  O?   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         AT   max         AT      ?  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??w?kP??   max       ??BZ?c?     ?  QX   
                     :            	      	      	      A   j            3         =   @   "      ?               P             A  	      "                           
                  N??oN*??N??5N>ԄN6?N??N ??O?'?N?M5O??-N?OSV?N?4lN-??N˝xOo??O?~MP?.PM?N?DN???N??P?O!?O.?PV&|P?UP?/N???PK?O?z?OM?8N???NQ??PS??O@8EO)??O??>O???P???O?R?O?Nq$?N$?ZO?^?O?#DO??N]??N?2N7(tNa??O?2N¿?O?xN?4?NOȡN?8?H?9???h??`B?????ě?;o;D??;??
;??
;?`B;?`B<o<o<#?
<T??<u<?o<?1<?1<?1<?j<???<???<???<???<???<??h<??h<??h<???=o=o=C?=C?=\)=?P=?w=@?=@?=@?=H?9=L??=P?`=]/=aG?=aG?=y?#=}??=?o=?+=?hs=???=??-=??T=???=???=???????????????????????_aaanpuuna__________????????????????????????

????????????????????????????????????????????????????????

 ???????????fadhuz???????????zmf,)./<AHQUaURH<1/,,,,?? 	"/3M\_^RH/	??????????????????????????????????????????Z[\gt???????ztgd^[ZZ????



???????????235@BN[be_[NB5222222YYcgt?????????tg`_`Y???????
#,&
?????????5N[gmmg`NB)????????????????????? 

?????????????????????????????lilmmz????????zmllll????????????????????<DHUanpuyz?zoaUNHB<<???????????????????????????? ???????????r?????????????????zr???)ADOWWQSO6)?NMLLOT[`hhklkh[WRONNw?????????????????zw???????????????#0<HLNNKIE<0+# ruyz???????zrrrrrrrr#")5?B=5,)##########,19;BN\h????????th6,?????? 
???????????????????????????-*/;HTX_gmrsoaTHC>3-???)/5:;:5/???05@????????gN)??????????????)6BEIIFB6)#	
)5=:5)





????????????????????efkt????????????unge??????
/8=;/#?????????????????????????????????????))

%)),,)#"#&*$#########????????????????????????????????????????JOOY[hkt|{tsh[OJJJJ{|?????????????????{?????

???????????????????????????#./45/,######?$?0?9?=?D?I?L?I?>?=?0?0?$?? ? ?$?$?$?$???????????????????????????????????????????????%??????????????????????????????????????????????????????????????ż???????????????????????????????????????????????????????????׺????????????a?n?o?r?p?n?g?a?U?I?U?`?a?a?a?a?a?a?a?a?O?[?tāĚĠĩīĥĚĕā?t?X?I???C?E?L?O????????????????????????????????????	?"?5?H?T?Y?Y?T?H?;?/??	????????????'?)?3?7?6?3?'??#?&?'?'?'?'?'?'?'?'?'?'?????????????????????????????????????????????????????????????????????????????????#?%?/?3?0?/?#??????#?#?#?#?#?#?#?#?b?n?s?{?{Ņņ?{?n?b?^?Y?W?_?b?b?b?b?b?b?N?Z?g?o?~?v?s?g?e?N?C?5?(? ?? ?(?5?A?N??(?5?<?<?0?.?#?????????????????????????$?G?b?f?=???Ƴƚ?x?o?jƎƚƧ?̹_?x?ù?? ????
???Ϲƹ??????????x?i?_?M?P?Z?f?q?r?f?]?Z?M?J?J?M?M?M?M?M?M?M?M?s?????????????????????????w?s?j?s?s?s?s?O?\?h?u?u?~?{?z?u?h?]?\?O?L?N?M?O?O?O?O?O?\?W?Y?U?F?*?????ż????????????6?H?O?B?N?R?[?`?]?Z?N?A?;?5?3?(?"?&?(?,?5?@?B?A?B?N?Z?d?g?s?{?s?g?_?Z?N?A?8?7?8?>???A??'?2?:?3?6?(????Ž?????????????н??ù??????	????????????ìâÔÍßâìù?.?;?T?_?d?`?T?G?9?"????׾þϾо׾???.????????????????????????s?l?k?s???????????'?-?.?)??????ֺɺ??????????ɺ??????(?5?H?Q?S?I?5????????ݿԿ޿޿???????(???????????????????????x?r?i?c?f?m?r?ŠŭŹŻ??žŹŭŠŞŘŖŠŠŠŠŠŠŠŠ?	?????	?????????	?	?	?	?	?	?	?	?	?	?лܼ?4?X?r?????????r?Y?4?????????˻лûлӻܻ??޻ܻлû????????????????????ÿG?T?`?c?m?p?u?p?m?`?_?T?G?C?;?8?9?;?B?G?<?I?U?b?c?[?U?I?<?/?#??
???????????#?<?#?<?H?L?I?C?9?0?#?
?????????????????
?#?[āĚĳĿĳ?t?K?A?2?.?$?$?!???!?+?B?[?y?????????????????????y?l?f?`?^?Z?`?l?y?S?\?W?M?B?:?!????????????????!?-?D?S?t?t?|?y?t?g?a?c?e?g?p?t?t?t?t?t?t?N?[?g?q?i?g?[?X?N?M?N?N?N?N?N?N?N?N?N?N??"?/?8?@?C?>?;?/?"?	????????????	???????	??????)?"??	??????????????????"?;?H?R?P?H?;?/?	?????????	?
?
????z?|?~?~?z?z?n?f?a?Z?a?d?n?x?z?z?z?z?z?z?@?@?<?@?G?L?W?Y?e?e?~?????~?r?f?e?Y?L?@?~?~???????????????????~?~?~?~?~?~?~?~?~???'?1?'?"????????????????????(?-?4?*???????????ݽٽݽ???????3???@?L?N?W?Y?\?Y?L?@?4?3?,?*?)?3?3?3?3?????????üм׼ּʼ???????????|?x??????E?E?E?E?E?E?E?E?E?E?EuEtEpEuEuEzE?E?E?E??
?????!???
?????
?
?
?
?
?
?
?
?O?C?A?6?*?"?????*?6?C?O?T?O?O?O?O?O = R ; T K , X   = : a e 1 * B 5 q Q P e e - V * < @ & D g ! i  2 & m ? ( L D ^ H Z k 2  ? R y A 6 j D [ \ ? j 9    ?  H  ?  r  ]  ?  B  ?  ?  @  Q      I     ?  ?  ]  ?  X  ?      q  ?  ?  ?  ?  ?  ?  ?  ?  ?  Z  F  ?  n  ?  -  ~  .  n  ?  G  ?  ?  ?  ?    ]  ?  o    9  
  ?  ݽ?w???????
?ě????
<T??<#?
=?%<?j=?P<#?
<?o<u<?t?<ě?<ě?<?/=? ?>J<ě?<?/<???=???=?P=t?=?-=?v?=}??=t?>%?T=aG?=u='??=#?
=??=y?#=e`B=???=?`B>??w=??P=?1=aG?=}??=??-=?{=? ?=??P=???=??=??T=Ƨ?=ȴ9=?`B=?"?=\=?x?B?1B??Bs?B ?B*5BB =uB4?B ?OB??A?w?B![B ?B	??BK?B?B	?hB??B$?BzWB$1B[@A???Be?BGB8?B"y?B4B=B@?B?CB?B%?GB 13BBjiB#rVB=?A???B??B?B-?B=?By?BR}B
??BCBҌB!?B?eB% /B?zB~?B??B?BhB?qB?B??B??BAvB6?B*9 B ??B??B ??B??A?owB!|?B ??B	zB??BWoB	?cB??BN>BAB$@ B?A??BA?BhBkB"?.B9?B?TB<?B??B>bB%??B ??B;?B@OB#??B#?A??B@5B??B-=?B??B??B@B
{?B{?B?B!=?BBB%#B?B??B?ZB??B??B?B?@B
Q?A?p?A?FA??J@??F@R??AƯ2AܡeA??A???????A?9|A??A?m?A?TA?C?A???B)?>IG?A???A?$?B?A?xA?n=A???A-??A?=A]??AG??@L	?A??5@???A?\ZA??|@?H@??Ag+A꘦A??A??A??@l??A?rA?ݱA???A?(?A???AǪ6?ଯ@?L@?x?A2??????@?0?C???A??B .9B
DEA?h?A?{?A???@?A?@S??Aƅ?A?{[A҄?A?????\?A??PA?A?s?A???A?? A?j?B+?>Ah$A??A?̺BjA??QA??A???A.qZA΁?A`??AG??@K??A??Y@??A?ZEA??v@?(?@???Ag?A??A??A??A:?@cA?A???A?|XA??GA?)?A???A?vt???@?@???A2????q?@???C???A???A???         	               ;            	      	      
      B   k            3         >   A   "      ?               Q         !   B  
      #      	                                                                      #                        A   7            +         5   )   +      '   )            7         !   %   =                  #   #                                                                                    =   !                        !   +                     5         !   #                     #   #                              N??N*??N??5N>ԄN6?N??N ??O?fN?G?O??.N?OSV?N?4lN-??N˝xOo??O?tWP?}?O??N?DN???N??O?Q?O??Nϋ?O!*?O?CP?BN???OPpOX??O:H?N???NQ??P=?'O,?O?O??>O?y?O?²N??AO?Nq$?N$?ZO?^?O?#DO??N???N?2N7(tN?y?O?2N¿?O?xN?4?NOȡN?8       ?  ?  ?  x  >  	?  ?  ?  V  ?  s  Z  ~  /  	  k  
?        }  ?  ?     ?  4  ?  ?  ?  =  /  ?  ?  x  5  =  	?  ?  ?  ?  ?  N  C  ?  B  ?    ?  g    ?  Y  ?  ?  ??D?????h??`B?????ě?;o;D??<?t?<o<?o;?`B<o<o<#?
<T??<u<?t?<?j=T??<?1<?j<???=?P<?/<??h=?o=?P=o<??h=?j=?w=C?=C?=C?=#?
=?w='??=@?=P?`>5?}=q??=L??=P?`=]/=aG?=aG?=y?#=?%=?o=?+=?t?=??-=??-=??T=???=???=???????????????????????_aaanpuuna__________????????????????????????

????????????????????????????????????????????????????????

 ???????????lmpu}???????????zwnl/-/<HMU[UMH<4///////	"/;HRTWVOH;/	????????????????????????????????????????Z[\gt???????ztgd^[ZZ????



???????????235@BN[be_[NB5222222YYcgt?????????tg`_`Y???????

??????????	)BN[ellf_NB)????????????????????????? 

?????????????????????????????lilmmz????????zmllll????????????????????D>EHUWamnswynlaURHDD???????????????????????????????????????????????????????????????:?OTSLOKB)?NMLLOT[`hhklkh[WRONN????????????????????????????????????????%0<EIKMMJID<0/#ruyz???????zrrrrrrrr#")5?B=5,)##########6;=BQYZh???????t[B66???????

??????????????????????????-*/;HTX_gmrsoaTHC>3-????)-59:85)???" "&)5BN[lswsi[NB5)"????????????????)6BEIIFB6)#	
)5=:5)





????????????????????efkt????????????unge??????
/8=;/#?????????????????????????????????????))

%)),,)#"#&*$#########????????????????????????????????????????JOOY[hkt|{tsh[OJJJJ{|?????????????????{?????

???????????????????????????#./45/,######?$?0?6?=?C?I?K?I?=?0?$??"?#?$?$?$?$?$?$???????????????????????????????????????????????%??????????????????????????????????????????????????????????????ż???????????????????????????????????????????????????????????׺????????????a?n?o?r?p?n?g?a?U?I?U?`?a?a?a?a?a?a?a?a?h?tāčĚġĤĞĚčā?t?g?[?U?K?M?O?[?h????
??	???????????????????????????????"?%?;?D?L?I?B?;?7?/?"?????? ?	???'?)?3?7?6?3?'??#?&?'?'?'?'?'?'?'?'?'?'?????????????????????????????????????????????????????????????????????????????????#?%?/?3?0?/?#??????#?#?#?#?#?#?#?#?b?n?s?{?{Ņņ?{?n?b?^?Y?W?_?b?b?b?b?b?b?N?Z?g?o?~?v?s?g?e?N?C?5?(? ?? ?(?5?A?N?(?5?8?7?5?.?+? ???????????????(Ƨ?????????3?R?[?]?=???Ƴƚ?z?q?mƎƚƧ???????ùܹ????????????ù??????????????M?P?Z?f?q?r?f?]?Z?M?J?J?M?M?M?M?M?M?M?M?s?????????????????????????w?s?j?s?s?s?s?O?\?h?u?u?~?{?z?u?h?]?\?O?L?N?M?O?O?O?O???????6?B?A?<?,?*???????????????????5?A?N?Q?Y?Z?_?Z?Y?N?A?5?*?(?#?'?(?/?5?5?N?W?Z?e?g?n?g?a?Z?W?N?H?A?>?=?=?A?I?N?N?ݽ??????????????ݽнĽ??Ľ̽н?ù??????????????????ìàÙÓØàæìù?.?;?T?a?b?T?G?2?"??????׾ҾԾ???????.????????????????????????s?l?k?s???????????????????????????ֺɺȺƺֺ̺??(?5?8?D?H?J?F?A?7?5?(????????%?(?????????????????????????{?r?j?e?f?q?ŠŭŹŻ??žŹŭŠŞŘŖŠŠŠŠŠŠŠŠ?	?????	?????????	?	?	?	?	?	?	?	?	?	????4?Q?r???????????r?Y?4???????һܻ??ûлڻܻ??ݻܻлǻû??????????????????ÿG?T?^?`?m?m?s?m?`?X?T?G?F???;?9?:?;?E?G?<?I?U?b?c?[?U?I?<?/?#??
???????????#?<?#?0?<?E?I?G?A?7?0??
???????????????
?#?O?[?h?t?zĈČĊĄ?t?h?[?O?B?9?6?7?9?A?O?y??????????????????y?w?l?j?f?g?l?n?y?y?S?\?W?M?B?:?!????????????????!?-?D?S?t?t?|?y?t?g?a?c?e?g?p?t?t?t?t?t?t?N?[?g?q?i?g?[?X?N?M?N?N?N?N?N?N?N?N?N?N??"?/?8?@?C?>?;?/?"?	????????????	???????	??????)?"??	??????????????????"?;?H?R?P?H?;?/?	?????????	?
?
????n?z?}?}?z?n?n?m?a?Z?a?e?n?n?n?n?n?n?n?n?@?@?<?@?G?L?W?Y?e?e?~?????~?r?f?e?Y?L?@?~?~???????????????????~?~?~?~?~?~?~?~?~???%? ??????????????????????%?(?,?3?(?????????????????? ??3???@?L?N?W?Y?\?Y?L?@?4?3?,?*?)?3?3?3?3?????????üм׼ּʼ???????????|?x??????E?E?E?E?E?E?E?E?E?E?EuEtEpEuEuEzE?E?E?E??
?????!???
?????
?
?
?
?
?
?
?
?O?C?A?6?*?"?????*?6?C?O?T?O?O?O?O?O 9 R ; T K , X ! 8 . a e 1 * B 5 b O ; e e - K $ > 1  F g  &  2 & l < $ L ?  A Z k 2  ? R L A 6 _ 5 [ \ ? j 9    ?  H  ?  r  ]  ?  B    ?  9  Q      I     ?    L  ?  X  ?    ?  0  ?  k  &  w  ?  ?  ?  ?  ?  Z  ?  p  9  ?  ?  i  #  n  ?  G  ?  ?  ?  [    ]  Q  ?    9  
  ?  ?  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT  AT           
     ?  ?  ?  ?  ?    ^  ;    ?  ?  ?  y  D     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  ]  I  6  #     ?   ?  ?  ?  ?  ?  ?  ?  ?  x  v  u  u  v  t  ^  G    ?  x  '   ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  r  f  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    {  w  x  f  W  N  F  =  *    ?  ?  ?  ?  ?  o  L  (  ?  ?  ?  l  >  G  O  T  R  P  L  H  C  5  %    ?  ?  ?  y  K    ?  ?  	H  	?  	?  	?  	?  	?  	?  	?  	m  	8  ?  ?  I  ?  S  ?     7  F  v  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  O  ,    ?  ?  k  2    ?  ?  J  l  ?  ?  ?  ?  ?  ?  ?  p  M  (  ?  ?  ?  U  ?  ?  3  ?  V  U  U  T  S  R  R  R  S  S  T  U  V  V  W  X  Y  Y  Z  [  ?  ?  ?  ?  ?  {  o  c  c  f  a  T  G  3    
  ?  ?  ?  ?  s  k  d  \  S  J  @  6  +      ?  ?  ?  ?  ?  ?  t  f  W  Z  _  e  f  d  _  Q  C  6  )      ?  ?  ?  ?  ?  f  D  #  ~  j  W  D  3  %      ?  ?  ?  ?  ?  ?  ?  Q    ?  ;  ?  /         	           
    #  "      ?  ?  ?  b      ?  ?        ?  ?  ?  ?  ?  u  Q  4    ?  ?  ^     ?  f  j  \  L  @  ;  4  !     ?  ?  `  	  ?  6  ?  ?  2  ?  E  ]  9  	U  	?  
n  
?  
?  
?  
j  
(  	?  	Y  ?  @  ?    ?  ?  ?          ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  `  G  .     ?            ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  }  x        ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  q  [  E  /       ?  I  ?  6  g  |  v  e  H  &  ?  ?  ?  J  ?  ?  >  ?  G  I  I  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  ]  F  0      ?  ?  ?  b  x  ?  ?  ?  ?  ?  ?  ?  ?  ?  t  `  K  6       ?  ?  ?  _  }  ?  Q  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  f  ?     G  {  u  ?  ?  ?  ?  ?  ?  ?  v  :  ?  ?  4  ?  Z  ?  8  I     Y  (  1  3  -      ?  ?  ?  ?  ?  ?  ?  l  -  ?    (  ?  b  ?  ?  ?  ?  ?  ?  ?  ?  {  \  9    ?  ?  ?  ?  r  P  +    

    ?  8  ?    U  ?  ?  ?  ?  G  ?  T  ?  
?  	?  ?  ?  m  ?    ?  ~  ?  ?  ?  ?  ?  ?  g  ?    ?  ?  ?  P  ?  7   ?  :  ;  :  4  *      ?  ?  ?  \  &  ?  ?  ?  ?  1  ?  ?     /      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  k  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  {  o  ^  N  7    ?  ?  ?  ?  n     ?  h    6  ?  ?  ?  ?  ?  ?    ?  }  ?  ?  y  r  w  x  v  p  g  Z  G  ,    ?  ?  k  0  ?  ?  U    ?  ?  /  3  5  3  *      ?  ?  ?  ?  v  Q  '  ?  ?  ?  ;  ?  ?  =    ?  ?  ?         ?  ?  |  :  ?  ?  L  ?  ?  <  ?  +  	u  	?  	?  	?  	?  	?  	a  	1  ?  ?  ?  a  "  ?    ?  4  =  .  $  ?  ?    ?  ?  ?  a    ?  ?  ?  o    M  ?  q  ?    :  m  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  s  9  ?  ?  m  @  ?  ?  ?  ?  q  Y  ?  G  _  >  
  ?  ?  {  B     ?  M  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  z  s  q  r  r  s  t  ?  ?  ?  ?  ?  N  K  H  >  1  #      ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    W  C  ?  0    ?  ?  ?  ?  c  C  #    ?  ?  ?  g  1  ?  ?    ?  ?  ?  ?  ?  v  W  8  ?  ?  ?  ?  j  0  ?  ?  :  ?  t  ?  B  3  ,  #  $    ?  ?  ?  ?  f  1  ?  ?  p  .  ?  ?  x  L  ?  ?  ?  ?  ?  ?  ?  ?  l  L  ,    ?  ?  ?  _  (  ?  ?  k      ?  ?  ?  ?  ^  :    ?  ?  ?  ?  ?  s  L    ?  b  ?  ?  ?  ?  ?  ?  ?  x  k  ^  E  *    ?  ?  ?  ?  ?  ?  ?  ?  [  b  e  Q  =  *        ?  ?  ?  q  6  ?  ?    ?  ?  ?  ?    ?  ?  ?  ?  ?  ?  ?  m  E    ?  ?  ?  z  G    ?  ?  ?  .  ?  ?  `    ?  ?  ?  U    ?  ?  R  ?  n  ?  ?  3  ?  Y  I  7      ?  ?  q  -    ?  ?  ?  y  G  ?  ?  ;  ?  x  ?  ?  ?  ?  i  P  0    ?  ?  t  5  ?  ?  P    ?  ?  ?  @  ?  ?  ?  ?  q  Y  ?  !    ?  ?  ?  *  ?  ?  g  2  ?  ?  ?  ?  d  %  	  ?  ?  ?  ?  ?  p  X  >  #    ?  ?  h  5     