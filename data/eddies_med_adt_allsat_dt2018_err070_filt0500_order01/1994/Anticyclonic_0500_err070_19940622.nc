CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�"��`A�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nc�   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       =���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F�z�G�     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?θQ�     max       @v�33334     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q            �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @��            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >&�y       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B3,g       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�u�   max       B3%�       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?!.�   max       C��H       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C��       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          j       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nc�   max       P���       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?۬q���       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       =�F       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�=p��
   max       @F�z�G�     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @v��Q�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P@           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͱ        max       @���           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?s   max         ?s       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0��   max       ?۬q���     P  X�            -   
   )         &   #                        4               	   B      	      [         
      '      &   .   	         	   
      	         
                     %                     j            9   :O��N�6N2:P	�BN��PMHkO�X)N�w6O`M�O��lN��[N�<�N���NT��NcJ�O��O-|�PM�O��N��Nc�N�R1N�aPQb�O���O��O@�P��NGbN�k�NO��N�\�O��OAUcP���P,-INb�O2�[N���N��N�0SNߤNq�eO{<hO��N���N¸ON��N�xO��OA��OB�PIjO4&RN�&N,��N�0HO
��N�Q�O�W/N�p�O)�OC��OĚ5O��޼��
�e`B�D���t��ě��o%   ;o;o;o;D��;D��;ě�;�`B;�`B<o<t�<#�
<#�
<#�
<#�
<49X<D��<T��<u<u<�o<�o<�t�<�t�<�t�<�9X<�j<�j<ě�<���<�`B<�<��=o=+=C�=C�=\)=\)=\)=\)=\)=t�=t�=��='�=0 �=D��=D��=L��=T��=Y�=e`B=m�h=u=�\)=�^5=ě�=���65<ATUbnwz���{nbUI<6�w�����������������������������������������#/4DHGKUQH/'$����� �������/HKbc^bZTH;/	���������������������"#)-5BN[[d[NFB5)""""B<=HUanv}����zqnaULBrllnv�������������tr �
#$#" 
      #*/2<?E<6/(#����������������������������������������-)(-07<HE?<0--------3457:?BNXWXXXUNB9533*&)(/<HMU]afa\UPH</*wsuz���������������w���������
�����)*5853)��������������������)*)(%	�����������������������DLX[OG6)��������)5BO[dgliaT5)��x{tv~��������������x[YZYYYamz������zvea[(5[t�������gB5��������������������$(/;<EHUZUHH</$$$$$$������������������������������������������������������������#%)58BEILUNB5)&!5[t���������{[NB>>PUaz���������z`UB��������������������^_dhlu|����������uh^��������������������YVZaentz}�}znaYYYYYY$)3166863)16646;BJO[f`_XNGB;61ptv����������tpppppp�������!��������������������).67666)% ")6765)���������������eggqtw���{tgeeeeeeeegkms�������������tgg�����������������������#06;640#
 ������������������|�����������������|#)/3:<=<</#��������������������)*))$#�������������������� ! ##/0<ILNIGG;30&# ������

����������������������������C@>AFHQTajhhfda`XTHC(*-169?ABCJ[cfg[O6)(����6B[horhO6���������
#$
�������ɻ��������лԻܻͻ��������x�t�h�m�t����������	����������������������������������������������������������������������m�����������������y�m�G�.����"�.�T�m����� �'����������������������H�a�m�����z�T�;�"�����������������	�!�HŭŹ����������ŹŭŠŔŇŀ�~łňŔŠţŭ�6�B�J�O�W�U�V�O�O�B�@�;�6�3�-�+�6�6�6�6F=FJFVF_FdFaFUFJF=F$FFFFFFF$F+F1F=�M�Z�f�s�������������f�Z�M�A�=�9�>�K�M����������������������������������������������������������������������������������������������������y�m�`�]�e�m�y����E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�r���������������r�o�l�r�r�r�r�r�r�r�r�ѿݿ�����������ݿѿĿ����Ŀɿѿ�������������
����������������������N�g�s�������������s�g�Z�H�5�0�,�+�/�7�N������(�5�J�N�R�W�N�5�����������Z�f�m�k�g�f�]�Z�P�M�F�J�M�R�Z�Z�Z�Z�Z�Z�����������������{�w�������������������������������������������y�x�w�y�~���������������'�'�,�'�������������龱�ʾ�;�S�Q�"����ʾ��������������������h�uƁƎƧƳƱƳƯƧƚƎƁ�u�h�U�P�Q�\�h�������������ýŽĽ��������������������Ƴ������������� ������������������ƳƯƳ��������"������������Z�N�I�L�e�s�{�������ĿſȿĿ�����������������������������������
�
�
�
�����������������������������������������������������������������ìù��������������ùìàÚààéìììì��������!�!�����������ôòù�������$�'�0�=�H�I�=�6�0�$��������������$�t¦¯®¥�g�)���������V�`�_�t�(�5�N�����������s�g�d�N�(����������(����&���������������������ʾ׾���������������׾ʾ��������������)�.�.�)�!������������������4�A�M�W�Z�]�Z�T�M�A�4�,�)�,�4�4�4�4�4�4���������������û��������������{�|�������лܻ������
��� �������ܻջлȻм'�.�4�>�@�D�D�@�4�2�'�%�!��'�'�'�'�'�'�������������������������������y�x�������"�/�;�H�K�P�T�T�N�M�;���������"��������������������������������������������������������������������������������(�4�A�M�Z�f�t����~�s�f�4�(������(�;�G�H�T�]�T�J�G�@�;�:�:�;�;�;�;�;�;�;�;���������������������������{�t�m�b�a�w��ÇÓ××ÓÏÎÌÇ�z�n�a�R�X�a�j�n�zÀÇ������!�'�$��������ؼּ޼������U�a�y�{�c�U�H�<�/�#��
������/�A�E�U�����(�4�7�@�4�3�(�������������Z�f�s�����������s�m�f�Z�Y�T�T�Z�Z�Z�Z����	��
�	�������������������#�0�4�<�G�A�<�0�.�#��
�	�
������������'�,�'��������ܹϹȹιϹܹ躽�ɺֺ���������غֺɺ�����������D�D�D�D�D�D�D�D�D�D�D�DwDrDwD{D�D�D�D�D��y���������������������y�r�l�g�b�l�t�y�y���
��#�0�3�<�B�I�<�0�#��
� ���������������������ɺֺ���ֺɺ���������������������������˼ͼü������������{�y�{�y�E�E�E�E�E�E�E�E�E�E�EuEqEpEiE\EPECEPEiE� / U i c 9 A / a l % 4 4 � < - S & $ A T G , K M G  ; Y N n R = @ D q ( G 2 ;  ; x j & ` j 9 f N @ }  [ . = G ] ` D ! 1 X J | V    Z  9  q  �    �    8  #  ]  �  �  z  w  m  @  r  �  v  �  7  �  �  �  }  L  �  �  F  �  x  �  }  �  O  �  k  v  �  �  �  e  �  �  �  �  8    G  �    �  �  }  �  7  �    '    �  �  �  �  λ��
�#�
�o=\);D��=�w<e`B<�o=#�
=�P<t�;�`B<D��<��
<u<��
=�P=}�=\)<�o<u<�o<�1=���=o<�j<�`B=��<�9X<�9X<�`B=@�=}�=49X=y�#=�hs=t�=,1=<j='�=0 �=<j=0 �=aG�=T��=8Q�=��=ix�=#�
=�7L=�C�=ix�=���=�%=u=]/=u=���=�7L>&�y=��w=ě�=�F>��>$�/B'��B'�BMB��B:�A���BB|BS BxeB�B�B�B��B%��B�B�B<SB�0BW�BBb�B!s=B�GB�BQcA���BE�Bf�B��B8�B!�ZBX�BV�B	!B�B�B3,gBFB��B�B�B��B��B�B(qB6�B#�hB	��B
�BT�B$�gBtB(DB �B#�BH�B�QB%��BcB,maA���BQ�B�=B��B']7B��B�5B7dBC�A�u�B�?BD�B@VB�+B��BH�B2�BW�B%� B)dBاB ��B�MBM�B��Bw4B!�	BȰB�B��A�u�B>�BI1B��B��B!��B=�B?�B
��B�wB�#B3%�B@B�B=�BC�B��B�BB|B@B?�B#A&B	��B?�B(AB$�$B��BD�B(&B"�B?�B��B&#B<�B,BYA��BD�B;�BZ^@�4n?IOA���Aj�aA���A�G�A��A�Y�C��HAB�A��+A�h"Am�0C�l�@�XA~EA�k�A�,�A���A?LA�'lApR�@�v�AS��B=�A!`�B��A���Aw
�A��eA�&&A�2A�3�B	6|A�E�A�b�A�W]AS�lA��A;z@��5@��f@� lA�g5A�5�A�5AA�i�A<;Ad�xAq-�A��.AҲA�f�A3�AB��AYLrA���?!.�@9;QC���A��A�� @#��@�ޛC� @�]�?O�bA��,Ak�A�vYA�kiA���A�}�C��AC�A���A��QAnG@C�h
@�*A�8A�YA�YA��<A?"�A�}WAp3G@�OAR��B�vA"��B�A���AwXA�+A��A͆�AїRB	?�A��rA���A�a�AR�RAԃ�A;�@�;�@��[@��A�x�A�A�qDA�o�A<'Ae bAs �AȃA)A�̩A2,BAB�@AZ�{A�?��@B�C���A�A�{�@ �?@��*C�            -   
   *         '   $               	         5               
   C      	      [               (      &   .   	         
         
                               %            	         j            9   :            +      /                                    '                  3   '         ?                     9   )                           #                        +                                 %   #                  '                                                         '                              9   '                           #                        )                                 #   Oh��N�6N2:OO�N�q9P�WO�X)N�O26�OQ�N��[N�<�NLM,NT��NnO��N�e�O��N��N}�Nc�N�CN�aO�NO���O��O@�ONGbN�k�NO��Nb�<O+�N��NP���P�Nb�O2�[N�G,N��N�0SNߤNq�eOA�O��N`FBN¸ON��N�xO�&NU��O3�O��hO4&RN�&N,��N�0HN�`�N�Q�OVUN�p�O)�OC��O�KiOH�q  a  B    ,  =  {  �  c  �  
  K  k  �  _  �  h  �  �    �  l    F  �  �  l  A  �    �  9  t  �    z  %  {  �      �  $  �  5  m  �  �  �      
F  a  �  ^  Z  �  �    �  �  ~  0  ?  �  
���C��e`B�D��<#�
��o<o%   ;��
;ě�<u;D��;D��<o;�`B<t�<o<��
<ě�<���<49X<#�
<D��<D��=0 �<u<u<�o=���<�t�<�t�<�t�<�h=�P<�/<ě�<�<�`B<�=+=o=+=C�=C�=��=\)=t�=\)=\)=t�=��=ix�=8Q�=<j=D��=D��=L��=T��=e`B=e`B=�j=u=�\)=�^5=��`=�F:9>GISUbnrv|~wnbUI<:�w��������������������������������������
#/8<>@=<:/#�����������"/;KQY[YYWTH/��������������������&&)05BNRVNBA5)&&&&&&D?BHUant|���zvneUOHD{wy����������������{ �
#$#" 
      #*/2<?E<6/(#����������������������������������������/+*0<BB<<0//////////3457:?BNXWXXXUNB9533////<HMUUUPH<///////y{��������������~{y����������������
')565/)





�������������������� &#��������������������)06ADC;6*���)5BO[dgliaT5)��x{tv~��������������x[YZYYYamz������zvea[5559BNX[dgkigd[NDB:5��������������������$(/;<EHUZUHH</$$$$$$������������������������������������������������������������% !()5@BEGNQNHB5+)%%5[t���������{[ND@@HUanz�������znaUD��������������������^_dhlu|����������uh^��������������������YVZaentz}�}znaYYYYYY$)3166863)16646;BJO[f`_XNGB;61ptv����������tpppppp�����������������������������)-66654)&!")6765)���������������eggqtw���{tgeeeeeeeehimou�������������th��������������������
#(010000,#
��������� �������|�����������������|#)/3:<=<</#��������������������)*))$#�������������������� ! ##/0<ILNIGG;30&# ������


���������������������������C@>AFHQTajhhfda`XTHC(*-169?ABCJ[cfg[O6)(��"6B[ahmnhXB6�����������

����ֻ����������ûǻϻ����������x�o�r�y����������	����������������������������������������������������������������������m�y�����������������y�`�T�K�G�@�F�T�`�m������"��������������������H�T�a�p�s�q�a�T�H�;�"�	������������"�HŭŹ����������ŹŭŠŔŇŀ�~łňŔŠţŭ�6�B�F�O�T�S�Q�O�B�?�7�6�/�1�6�6�6�6�6�6F=FJFVF[FaF^FSFJF=F1F$FFFFF$F-F1F8F=�Z�f�s��������������s�f�Z�S�N�M�L�M�T�Z������������������������������������������������������������������������������m�y�����������y�w�m�`�`�`�h�m�m�m�m�m�mE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�r�����������r�r�p�r�r�r�r�r�r�r�r�r�r�ѿݿ�����������ݿѿĿ����Ŀɿѿ�������������������������������������g�s�����������������s�g�Z�?�9�9�B�N�Z�g���(�-�5�6�5�+�(������������Z�f�k�j�f�e�[�Z�V�M�H�M�M�T�Z�Z�Z�Z�Z�Z�����������������{�w���������������������������������������{�y�������������������������'�'�,�'�������������龾�ʾ�������	����׾ʾ��������������h�uƁƎƧƳƱƳƯƧƚƎƁ�u�h�U�P�Q�\�h�������������ýŽĽ��������������������Ƴ������������� ������������������ƳƯƳ�������������������������������������������ĿſȿĿ�����������������������������������
�
�
�
�����������������������������������������������������������������ùÿ��������ùìàÞàçìöùùùùùù������������������������������������$�0�3�8�0�.�$����������������t¦¯®¥�g�)���������V�`�_�t�(�5�N�g�������|�g�]�N�(������� ���(����&���������������������ʾ׾���������������׾ʾ��������������)�,�,�)���������	�������4�A�M�W�Z�]�Z�T�M�A�4�,�)�,�4�4�4�4�4�4���������������û��������������{�|�������лܻ������
��� �������ܻջлȻм'�.�4�>�@�D�D�@�4�2�'�%�!��'�'�'�'�'�'�������������������������������}�~�������"�/�;�H�K�P�T�T�N�M�;���������"���������������������������������������������������������������������������������(�4�A�M�Z�f�t����~�s�f�4�(������(�;�G�H�T�]�T�J�G�@�;�:�:�;�;�;�;�;�;�;�;�y���������������������������}�w�m�h�h�y�zÇÊÇÅ��z�n�a�`�a�n�n�w�z�z�z�z�z�z�������������������ܼۼ����U�a�v�x�n�h�a�R�<�/�#��
�����#�,�E�U�����(�4�7�@�4�3�(�������������Z�f�s�����������s�m�f�Z�Y�T�T�Z�Z�Z�Z����	��
�	�������������������#�0�4�<�G�A�<�0�.�#��
�	�
������ܹ����	�����������ܹϹʹϹйܹܺ��ɺֺ���������غֺɺ�����������D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D�D�D�D�D��y���������������������y�r�l�g�b�l�t�y�y���
��#�0�3�<�B�I�<�0�#��
� ���������������������ɺֺ���ֺɺ��������������������������ɼ˼������������������}�{���EuE�E�E�E�E�E�E�E�E�E�E�E�EzEuEuEsEmEpEu + U i 2 C C / W ^  4 4 � < , S % - - Y G  K * G  ; ( N n R ? = 6 q # G 2 K  ; x j . ` Y 9 f N > w  ] . = G ] > D  1 X J u O    �  9  q  �  �  �    �  �  ?  �  �  �  w  0  @  �  �  �  �  7  �  �  7  }  L  �  O  F  �  x  u  [    O  �  k  v  �  �  �  e  �  �  �  _  8    G  `  �      }  �  7  �  �  '  0  �  �  �  (  �  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  ?s  .  D  S  \  `  _  X  M  ?  /      �  �  �  �  Q     �   �  B  3  %      �  �  �  �  �  �  �  �  t  _  M  ;  )                
      �  �       "  2  C  H  H  I  I  I  �  &  K  n  �  �  �    )  $    �  �  �    v  �  P  �  �  7  8  9  :  <  3  (      �  �  �  r  F  0    �  �  H    �    F  g  x  y  i  H    �  �  2  �  �  W    �  s  �  &  �  �  �  �  �  �  �  �  h  I  (    �  �  �  �  k  J  %  �  C  V  `  c  c  a  `  ^  [  O  ;  !  �  �  �  p    y  �   �  �  �  �  �  �  �  �  �  �  k  &  �  ~    �    b  d  S  /  �  �  �  �  �  �    
    �  �  �  �  f     �  O  �  �  �  K  ?  3  '                  �  �  �             k  c  \  T  M  D  9  -  "    	  �  �  �  �  �  �  �  u  ^  n  ~  �  �  �  �  �  �  �  w  a  I  2      �  �  z  .   �  _  Q  P  [  \  Z  V  Q  H  <  ,      �  �  �  �  �  h  =  �  �  �  �  �  �  �  �  �  �  �  �  �  z  k  X  E  )  �  �  h  ]  P  >  '    �  �  �  �  �  v  _  I  5      �  �  �  �     F  i  �  �  �  �  �  �  �  u  ]  ?    �  �  j  /  9  �  '  [  �  �  �  �  �  �  �  o  Z  ;    �  M  �  �  &  �  W  j  w  �  �  �  �  �  �  �  �  �    �  �  �  O  �  <  �  �  �  �  �  �  �  �  �  }  v  n  f  ^  I    �  �  �  d  7  l  i  g  d  c  i  n  s  w  y  {  |  x  p  h  `  W  N  E  <  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  t  h  F  (  
  �  �  �  �  �    
  	          �  �  �  �  �  )  I  o  �  �  �  �  �  �  �  �  �  ~  L    �  l  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  O  "  �  �  �  l  k  j  d  [  Q  D  7  (    
  �  �  �  �  �  �  �  �  �  A  <  4  (      �  �  �  �  �  �  v  S  .    �  �  �  �    M  t  �  �  y  m  �  "  A  _  u  �  �  Z    �  �  g  ~      �  �  �  �  �  �  �  �  �  �  �  z  n  a  S  E  7  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  3  .  0  1  -  (  !        �  �  �  �  �  �  �  �         8  L  `  q  t  i  Z  G  1    �  �  �  x  f  �  U  �  �  �  3  W  m  �  �  �  �  �  �  b  +  �  �  C  �  �  t  �  �  �  �    
    �  �  �  �  �  h  <  	  �  �  W    �  �  z  S  +     �  �  i  P  0    �  �  �  �  s  9  �  �  �  �      %    	  �  �  �  S    �  �  M    �  v  -  �  ^  �  {  w  r  k  c  Z  Q  H  ?  5  -  '           	        �  �  �  �  �  �  �  �  �  g  N  5    �  �  �  �  x  H  
  �  �            �  �  �  �  �  x  g  V  K  C  M  �          �  �  �  �  �  �  �  �  �  d  A    �  �  �  h  6  �  �  x  _  F  0     #    �  �  �  �  p  L  *  !      �  $              �  �  �  �    L    �  �  E  �  �  _  �  �  �  �  �  �  �  �  �  �  �  �  ~  Z  4    �  �  <  �  -  3  5  4  *      �  �  �  �  �  v  F    �  �  |  5  �  m  b  U  J  @  9  4  -    �  �  �    \  -  �  �  �  �  l  F  q  �  �  �  �  {  R  &  �  �  �  [  %  �  �  �  U  <  &  �  �  �  �  �  �  �  }  x  r  l  f  `  Z  Q  I  A  8  0  (  �  w  e  R  ?  *    �  �  �  �  5  �  m     �  4  �  o      {  x  u  r  h  I  +    �  �  �  �  �  j  M  0     �   �        �  �  �  �  Y  %  �  �  d  ,  �  �  h    �    U  �  �  4  �  	5  	z  	�  	�  
  
  
+  
4  
E  
,  	�  �  �  �  �  �  R  \  `  a  a  _  W  L  =  .    	  �  �  �  �  Y     �  ~  �  �  �  �  �  �  �  o  A    �  �  S  �  �  �  X  �  }  �  ^  D  3  &       �  �  �  �  �  j  E    �  �  �  >  �  �  Z  J  8  &    �  �  �  �  �  j  M  3    �  �  �  �  �  C  �  �  �  �  �  �  �  s  d  U  A  )     �   �   �   �   �   �   v  �  �  �  �  �  �  �  �  �  �  �  �  �  z  s  j  a  X  O  E  r  }      �  �  �  �  �  g  ;    �  �  Z    �  {  �  e  �  �  �  �  �  }  a  E  #    �  �  �  �  p  T  9    #  2  y  �  L  �  �  �  �  �  �  x    �  �  (  c  �  �  
7  �  �  ~  y  r  i  ]  N  <  #    �  �  �  p  <    �  s    �  0  0  -  '  !      �  �  �  d    �  ~  &  �  `  �  �  �  A  ?  .  -  4  *      �  �  �  �  g  $  �  �  :  �  Y  �  �  �  �  �  �  V    
�  
�  
N  
  	�  	V  �  X  �  �  �  �  �  |  
�  
c  
8  
�  
�  
�  
�  
�  
�  
E  	�  	�  	X  �  j  �  �  �  J  �