CDF       
      obs    2   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       ?Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM????   
add_offset               min       ?h?t?j~?   max       ???\(?      ?  ?   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M?v?   max       Pˑ?      ?  t   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ??t?   max       =???      ?  <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?
=p??
   max       @FAG?z?     ?      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ???Q??    max       @v?
=p??     ?  '?   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @'         max       @N?           d  /?   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @?R        max       @?1?          ?  0   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ??`B   max       >???      ?  0?   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A?ߴ   max       B0|?      ?  1?   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A?^?   max       B0??      ?  2`   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ??c?   max       C??&      ?  3(   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       <?Q   max       C???      ?  3?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         	      ?  4?   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      ?  5?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      ?  6H   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M?v?   max       P??p      ?  7   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6??C-   
add_offset               min       ??7??3?   max       ??z????      ?  7?   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ??C?   max       >C??      ?  8?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ??z?G?{   
add_offset               min       @?
=p??
   max       @FAG?z?     ?  9h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ??z?G?{   
add_offset        @f?        min       ??\(??   max       @v?
=p??     ?  A8   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ??         
add_offset               min       @         max       @N?           d  I   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @?R        max       @??           ?  Il   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Av   max         Av      ?  J4   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6??C-   
add_offset               min       ??ݗ?+j?   max       ??xl"h	?     ?  J?      >         r                             	            I      U      (         
   d   5   E            ?   D      *   
      
   	      !         s      	         N???Pˑ?O?N??Ph?TO??NC?6O?iN???O??)O?n?NNFO???N?V?P???O?&Ot?N??P~N??sPK?N?RJO?&O&ĨO?uO'j?PW??Pi?}Pl?N??<O?	N??P:K?O???N?:O8??O??OOfN?.?NZi
Oz??O??O??N.??O???O?N9?N?"?N?grM?v???t?%   ;D??;?`B<t?<t?<#?
<49X<T??<e`B<u<?o<?t?<???<?1<?9X<ě?<???<?/<?`B=o=+=C?=\)=\)=\)=??=?w=?w=#?
='??='??=,1=,1=8Q?=<j=@?=@?=P?`=]/=aG?=aG?=q??=?C?=?hs=?hs=???=??-=??
=???fgnt???????tjgffffff??????)1BNXP5?????4-,15BNU[_a`[ZNBA544??????????????????????????5BJKH=5???????????

????????????????????????MFCN[gtt????xtg[WNMM?
#$/13/(#
??????????????????????????????)8;@@6???????????????????????????????????????????????????????????????
	!)5Ng???????tgB
!).6BCOQUVXWOB?60)}|?????????????????}??????????????FFM`hrx????????xt[OF/./4<HJUUXUKH<5/////???
#/<?&47.%
??????????????????????????????????????????????????????????????)6BYRVXTB6)?*6CO[\cb\SOC6*??????????????????)5DFSWWTJB)??)BNg???????{l[D5?????

???????????? ????????????

????????
#/=OTPH.#???????#/7=@?<6/#
??+068BOP[hjmhg[VOB6++??????
!
???????

???????
#&02/)#
??????????????????????????????????????????????)+*
??????????????????????????????????????????? ?
#%#"
        ??????$.31'?????????????????????????????  "'/;AEF@;/#"       #%&'$#
	
}?????????n?o?z?}?}?z?n?g?a?`?V?V?a?j?n?n?n?n?n?n???(?N?Y?_?^?Q?:?(????ѿ????????Ŀο??n?{ŇŔŗŞśŔŐŇ?{?n?b?b?Y?^?b?l?n?n?h?tāāąā?t?m?h?g?h?h?h?h?h?h?h?h?h?h??????<?U?\?\?X?J?<?#?
????????Ŀ???????H?U?n?zÇÌ?z?{ÇÄ?z?a?T?H?<?5?-?.?<?H?????????????r?k?r?t?}????????????????????????????????????????????????????????????????????????????(?5?=?5?5?/?(?!??????????????????"?.?;?Y?`?f?i?e?]?G?;?.?"?????????ìù????????????ùñìãìììììììì???????????????????????q?d?a?c?h?k?s??G?T?`?h?l?m?n?m?`?Y?T?G?D?F?G?G?G?G?G?G??6?[āđĘĚĔĊ?{?h?B?6? ???????????????????ľ˾ʾþ????????????????????????4?A?M?O?N?L?A?4?(?????????????(?4???????????ľ????????????????????????????4?f?r?????????????r?Y?M?4??????????4????????
?????????????????????????????g®¬¯®¡¤?g?B?)????!?5?N?g???????ûлӻܻ??ܻлȻû???????????????DoD{D?D?D?D?D?D?D?D?D?D?D{DoDkDbD^DbDgDo?5?A?I?N?Q?X?Y?[?_?Z?N?F?A?3?(?&?$?(?1?5?S?_?x???x?X?Q?A?7?-?!?????޺?????9?S?m?y?????????????????y?m?g?`?_?`?`?b?i?m?@?M?r?????ʼּ????????ʼ??????|?f?A?7?@Ƴ?????$?=?F?J?=?$??????ƧƎƄ?w?yƀƚƳ???"?=?M?Q?K?4???????????????????????????????ʾվ׾??????????׾оʾ???????????????????!?(?0?3?(??????????ݽֽݽ????A?M?Z?f?r?s???????s?f?d?Z?R?M?A?;?A?A???	??"?a?o?w?u?h?;??	????????????????D?D?EEE$E$EEED?D?D?D?D?D?D?D?D?D?D????
????"?#?'?%?#???
? ????????????EuE?E?E?E?E?E?E?E?E?E?E?E?EuEjEiEdEiEmEu?׾??????	??"?*?'??	?????????׾ʾǾǾ׿`?m?y?????????|?y?m?`?W?G?>?;?1?E?G?T?`?4?A?B?M?U?T?M?A?4?.?(?(?(?0?4?4?4?4?4?4??????????????????????????????????????????????
????????¿²¦¦¯¿??????F1FJFVFgFoFxF|F|FrFcFVF=FFFE?E?FFF1ÇÓàù??????????????ùàÓÉÆÇÈÅÇǡǭǱǭǫǣǡǖǔǓǔǛǡǡǡǡǡǡǡǡ?~?????????º????????~?e?R?@?4?9?C?V?r?~?	??"?/?/?/?"???	???????????????????	?_?l?x???????????~?x?u?l?_?_?_?_?_?_?_?_??????
???
???????????????????????????\?O?J?C?6?*?????)?*?6?C?O?Y?\?\?\?\?Y?e?r?x?|?r?e?]?Y?W?Y?Y?Y?Y?Y?Y?Y?Y?Y?Y : / % J  9 J A Z *   # 6 E ( 7 : 2 Z % ) M M Y v [ \ O [ L J s D ) f / z 9 ) l m Z " i , 4 _   P I  ?  l  K  .  ?    _  >  ?    ?  Z  X  ?  ?  >  ?  ?    ?  W  4  J  ?  ?  ?    6    ?  Y  ?  p    <  ?  ?  ?  ?  ?  ?  ?  N  g  /  Q  y  ?    "??`B=}??<u<t?=??#=t?<e`B=+<??
=o=,1<?9X=49X<?9X>???=+=,1=o=???=,1==H?9=??P=L??=y?#=8Q?>I?=?j=?/=49X=?7L=H?9=???=?;d=e`B=?9X=e`B=??w=y?#=?%=??
=?9X=?9X=??>=p?=? ?=???=?Q?=\=?/B
?BÜB??B ??B??BN~B!ŸB	(?BVB?qBZ?B!T?B=@B?B?zB?B??BQ?B B?MB=B#4]B??B`B??B0|?B?#B?-B?}B$<?B#J?B$.^B?Bz?B??B??B?B?B?-B*RB?wBj?B, B?B??B?B,?MA?ߴBEB??B	?fBIiB=?B ??B.?BE?B!??B	;]B??B?)BN?B!??BA?B??B??B??B?*BydB@VB?B@B#>?B?7B>?B??B0??B;?B<?B	>B$:#B#??B$@B?xB?B<XB?B?B>?B??B@BDBƺB^?B=?B??B?B,??A?^?BC?B?AA??A?37A??A܏SA??IA?,+@???A??A?A??bAa?A???AF@?Ag??Aى?AK~?A6z?AK?d@??QA???A?mN@?^rC??	A???@t?Am@?BO?A??AP??A1?A@lLA?"lC?M?A??RC??AXI?AiϽA:?@K?*A??C??&A̷?B«@?eA??@?=?A?lB i2??c?A?j?A??qA?~WA???A菠A?|?@?$?A??ZA?xA?}Aa?	A??0AE??AgxxAم?AL??A8x^AK??@?YdA?} A??G@?? C???A?y[@fT[Am )@??2B#hA??EAQ̷A0?~AA?A?~?C?H?A?}?C???AWؕAi?A;?@LwA??6<?QA?pB??@c?A?H?@?*?A?EpB ?????      ?         r                             	            J      V      )            d   6   F         	   ?   D      +   
          
      "         s      	         	      A         +                  #            9            1      /            )      5   3   ;            /                        !   !         %                     3                                                         !            )      %   '               /                           !                        N?J?P??pO?N??O?'?N?"NC?6N???N???O/?KO?x?NNFO]??N?V?O\??O?&OO??N??O^?$N???O?|N?}?N?
?O&ĨO?uO'j?O?P?O???N??<N??7N??P0??OD3?N?~rO A?O??O/s?N?.?NZi
O]??O??O?WWN.??O?$2O?N9?N?"?N?grM?v?  ?  ?      
?  ?  ?  ?  ?  ?  ?  ?  ?  ?    ?  ?  ?  ?  ?  P  ?    ?  ?  ?  
?  ?  ?  ?  ?  +  ?  d  W  ?  ?  ?  ?  ?  \  #  ?  ?  |  ?  q  K  $  ???C?<u;D??;?`B=}??<??
<#?
<?C?<T??<???<???<?o<?j<???>C??<?9X<???<???=m?h<??h=?o=C?=?P=\)=\)=\)=?hs=]/=?%=#?
=49X='??=49X=Y?=<j=H?9=@?=L??=P?`=]/=ix?=aG?=u=?C?=?/=?hs=???=??-=??
=???kgpt???????tkkkkkkkk?????)>@;5???????4-,15BNU[_a`[ZNBA544?????????????????????????+48985)???????


?????????????????????????????ROW[agt{ztog_[RRRRRR?
#$/13/(#
???????????????????????????????)36;:6)??????????????????????????????????????????????????????????????;:;?BN[govwutpg[NB?!).6BCOQUVXWOB?60)???????????????????????????????VSUWX[ht|???????th[V//6<HITUVUHG<50/////??????
"#"
??????????????? ??????????????????????????????????????????????)6BYRVXTB6)?*6CO[\cb\SOC6*???????????????????)5BFJONLB5)
""&3N[gt?????tgNB5)"?????

???????????? ?????????????

????????
#/<MQ@*#?????? 
#'/39<;//#
 ?226;BOR[hilhe[SOB622?????
 
????????

???????
#$-/0/,&#
??????????????????????????????????????????????'*)' ??????????????????????????????????????????? ?
#%#"
        ???????"#!????????????????????????????  "'/;AEF@;/#"       #%&'$#
	
}?????????a?n?z?}?|?z?n?c?a?a?W?W?a?a?a?a?a?a?a?a??5?H?M?P?G?-?????ѿƿ????????ѿݿ???n?{ŇŔŗŞśŔŐŇ?{?n?b?b?Y?^?b?l?n?n?h?tāāąā?t?m?h?g?h?h?h?h?h?h?h?h?h?h???
??#?.?9?<?:?0?#??
?????????????????U?a?n?o?r?v?n?a?U?H?G???H?L?U?U?U?U?U?U?????????????r?k?r?t?}???????????????????????????????????????????????????????????????????????????????(?-?/?0?)?(?????????????????	??"?.?;?G?Q?`?e?_?X?G?;?.?"??	??? ?	ìù????????????ùñìãìììììììì?s?????????????????????s?m?f?e?f?l?o?s?G?T?`?h?l?m?n?m?`?Y?T?G?D?F?G?G?G?G?G?G?6?B?O?[?h?l?u?u?n?h?[?O?B?=?6?-?+?,?3?6?????????ľ˾ʾþ????????????????????????4?A?F?M?L?J?A?4?(?????????????(?4???????????ľ????????????????????????????'?4?@?M?Y?f?u?y?r?f?Y?M?@?4?'?#????'??????	???????????????????????????????[?g?t?u?g?[?N?B?5?)?&?&?,?7?N?[???????ûлѻۻлǻû???????????????????D{D?D?D?D?D?D?D?D?D?D{DoDmDcDoDpD{D{D{D{?5?A?I?N?Q?X?Y?[?_?Z?N?F?A?3?(?&?$?(?1?5?S?_?x???x?X?Q?A?7?-?!?????޺?????9?S?m?y?????????????????y?m?g?`?_?`?`?b?i?m???????????ʼּ??????ʼ????????|?q?g?r??ƧƳ??????????????ƳƧƒƉƅƉƓƚƧ?	??"?/?A?B?>?6?%??	?????????????????	?????ʾվ׾??????????׾оʾ????????????????????(?,?(??????????ݽڽݽ??????A?M?Z?f?r?s???????s?f?d?Z?R?M?A?;?A?A???	??"?a?m?u?t?g?;??	????????????????D?EEEEEE EEEED?D?D?D?D?D?D?D?D????
????!?#?&?$?#???
?????????????E?E?E?E?E?E?E?E?E?E?E?E?EuElEiEeEiEjEuE??׾??????	??"?*?'??	?????????׾ʾǾǾ׿`?m?y?????????z?y?m?e?`?]?T?G?B?<?G?J?`?4?A?B?M?U?T?M?A?4?.?(?(?(?0?4?4?4?4?4?4?????????????????????????????????????????????
???
??????¿·²¦²¿????????F1FJFVFgFoFxF|F|FrFcFVF=FFFE?E?FFF1Óàù??????????????ùìàÓÊÇÈÊÇÓǡǭǱǭǫǣǡǖǔǓǔǛǡǡǡǡǡǡǡǡ?r?~?????????????????????~?r?e?Y?Q?U?g?r?	??"?/?/?/?"???	???????????????????	?_?l?x???????????~?x?u?l?_?_?_?_?_?_?_?_??????
???
???????????????????????????\?O?J?C?6?*?????)?*?6?C?O?Y?\?\?\?\?Y?e?r?x?|?r?e?]?Y?W?Y?Y?Y?Y?Y?Y?Y?Y?Y?Y * , % J  + J 3 Z &  # + E  7 7 2 8 & " I M Y v [ V 5 5 L I s C   c 0 z 7 ) l h Z   i   4 _   P I  ?    K  .  [  ?  _  ?  ?  v  i  Z  ?  ?  ?  >  ?  ?  ?  ?  }       ?  ?  ?  L  ?  ?  ?     ?  D  ?    m  ?  ?  ?  ?  ?  ?  1  g  ?  Q  y  ?    "  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  Av  r  ?  ?  ?  ?  ?  ?  u  `  H  +    ?  ?  ?  ?  m  C  ?  ?    ^  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  i    ?  Q  ?  A   ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  l  G    ?  ?  {  H    ?         ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  	b  	?  
   
M  
q  
?  
?  
?  
?  
}  
*  	?  	-  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?    5  =  ?  <  6  ,    ?  ?  ?  \     ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  u  N  $  ?  ?  ?  ]  P  ?  ?  ?  ?  ?  ?  x  h  X  B  *    ?  ?  ?  ?  ?  ?  |  p  |  |    ?  ?  ?  ?    m  V  9    ?  ?  ?  ?  n  P  +    ?  ?  ?  ?  ?  ?  ?  }  k  Y  F  1    ?  ?  ?  f    ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?              $  )  ?  ?  ?  ?  ?  ?  ?  ?  ?  y  L    ?  ?  }  ;  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  |  s  n  l    q  ?  W  ?  L  ?  Q  ?  ?       t  q  ?  M  
?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  o  U  6    ?  ?  s  1   ?   l  ?  ?  ?  ?  ?  ~  o  _  P  =  %    ?  ?  ?  m  6  ?  ?    ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  v  f  Q  5    ?  ?  ?  ?  ?  ?  @  ?    a  ?  ?  ?  ?  p  &  ?  ?  :  ?  ?    n  ?  s  }  ?  ~  z  s  h  Z  M  =  *    ?  ?  ?  ?  }  f  M  )  ?  ?    i  ?    =  P  B    ?  ?  u  G    ?  H  ~  >  P  ?  ?  ?  ?  ?  ?  ?  b  <    ?  ?  |  B    ?  h    ?  @  ?  ?  ?  ?  ?  e  +  ?  ?  S    
?  
W  	?  	y  	  ?         ?  ?  ?  ?  w  b  F  %    ?  ?  ?  d  =    ?  ?    ?   ?  ?  ?  ?  o  R  ,  ?  ?  u  U  r  C  ?    ?  z  K  ?  ?  |  ?  v  b  S  C  3  #      ?  ?  ?  ?  ?  ?  ?  ?  ?  ~  w  	  	|  	?  
  
1  
^  
?  
}  
B  	?  	?  	   ?    ?  -  ?  ?  ?  C  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  `  1  ?  ?  {  '  ?    q  c  ?  *  H  [  ?  ?  ?  ?  ?  ?  O    ?  ?  B  ?  \  ?  v  -  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  w  d  P  D  ;  2  )     ?  ?  ?  ?  ?  ?  ?  ?  ^  $  ?  }  )  ?  o    ?  
  ?   ?  +  %        
    ?  ?  ?  ?  |  R    ?  ?  e  -     ?  ?  ?  ?  ?  ?  n  E  (      ?  ?  f  ?  s     ?  <  ?  ?    <  \  d  a  P  :        
?  
?  
v  
#  	?  	
  A  g      c  T  U  V  R  G  8    ?  ?  ?  ?  a  ?  +    ?  ?  C   ?   ?  ?  ?  ?  ?  ?  ?  ?  j  B    ?  ?  g  ?  h  ?  (  p  ?  y  ?  ?  ?  ?  ?  ?  n  ^  O  @  /    ?  ?  ?  c    ?  ?  >    2  ?  >  ;  6  3  +  $    ?  ?  ?  ;  ?  }    ?  ?  ?  ?  ?  z  f  S  ?  *      ?  ?  ?  ?  L    ?  Y    ?  X  ?  ?  ?  u  _  H  1    ?  ?  ?  {  >     ?    <  ?  ?  k  $  P  P  3    ?  ?  ?  j  M  5  "    ?  ?  3  ?  t  Y   ?  #        ?  ?  ?  t  O  &  ?  ?  c    ?  -  ?  ?  ?  O  ?  |  k  W  @  %    ?  ?  ?  q  .  ?  ?  ]    ?  ?  7  ?  ?  ?  ?  ?  ~  g  O  7       ?  ?  ?  ?  Q      ?   ?   ?   a  ?  w  ?  '  Z  u  |  r  O    ?  J  ?    L  
^  	F    ?  )  ?  ?  }  `  B  !  ?  ?  ?  ?  k  B  !    ?  ?  ?  ?  }  #  q  h  ^  U  L  B  7  ,        ?  ?  ?  ?  ?  ?  ?  ?  y  K  )    ?  ?  ?  ?  ?  ?  k  P  3    ?  ?  ?  ?  ?  ?  ?  $          ?  ?  ?  ?  ?  K    ?  }  0  ?  }    ?  ?  ?  ?  ?  ?  ?  ?  ?  }  f  N  5      ?  ?  ?  ?  t  3  ?