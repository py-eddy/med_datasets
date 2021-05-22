CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��E����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       P�{�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �o   max       =�F      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F��
=p�     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @v��\(��     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @Q�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�F        max       @�:�          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >%�T      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��/   max       B3-T      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��o   max       B3E�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?0&a   max       C��d      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?"��   max       C���      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          N      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       P��      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���'RTa   max       ?ݡ���o      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �o   max       =�F      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F��
=p�     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @v�=p��
     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�F        max       @�'�          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?m   max         ?m      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?ݞ��%��     �  T         	                  $            ?                              3   
      -   '   '         !      -                        
   	            '            
         $      M   '      
      &NK7�N��#O�N�sN0��N��9Nn]�O�a�Oѯ�N���O��?N�<eP�{�NTN���N�<�P;tNLQN�_YO�r	O#	GNZZEPWlzOYg�P�4P@�O�zO�"'Ov��P��O� 6N��BP\F�O{�N��N�
Op��OēN���O��N�B-Na�]N��N���O��'O�_Oj��N�%OO�QNd]6OW1�N�jbOR�oN�O1: O](?O�yNL�}O)VO�	B�o��/��j��C��T���T���#�
�o�ě����
���
�D���o��o;o;D��;�`B<o<D��<T��<e`B<�o<�o<��
<�1<�1<�1<�9X<�j<�j<�j<�h<�=\)=\)=�P=�P=#�
=#�
=<j=H�9=H�9=H�9=L��=P�`=P�`=P�`=q��=q��=q��=u=y�#=��=���=�{=�-=�-=��=��=�F��������������������mllmnz������znmmmmmm������������������������������������������������������������`]^bnw{���{nmb``````��������������������' !)2;BN[chfb^[EB5'����
#/9?CH</'������������ �������)5?N[eie[SNB5) ''')05BNUNLBB;5)''''"+FXg������������gB"^ht{����ttsh^^^^^^^^��������������������!#&/8<<EF</#����������
��������������������������-*,/0<HSU\UUHF</----8=@AHUVan������znbH8xz���������������zzx�)-)#����������! )BZX]ZOB)��mprz�������������zqm����,5BQcgsqg[4��
 <Haq{{aU5#
:758<HUZadjja`UHB<::FJUanz}��������n]UIF! "/;HRTUVUTNH;/.($!Sgt������������xk_]S���������
����������������������������/-+��������������������������hhuu��������yuihhhhh_chntvtrtuttth______�����������������������')*)$!��	$)*,)&)16::<<65*&��������������������@?ABFNO[_][VQODB@@@@XWZ[htxtph[XXXXXXXX���� �����������������������������3//35BN[bgqx~|g[NB=3������������������������������������������������������������-05<IUTID<10--------:89?HU]ahmttojdTHA;:T]ajmnmlmnmca_UTRTTTiinp�������������{ni������������������������

�����SIORU^anz�����~zngaS'()/37BO[bc][OHB60)'����������������������������

������������68?=2)�������������������������������������������ؾʾ׾��������׾оʾ��þʾʾʾʾʾ��$�0�=�>�I�Q�S�U�J�I�=�;�0�+�%�$��!�$�$�t�w�x�z�t�h�d�a�h�n�t�t�t�t�t�t�t�t�t�t���������������������������������������������������������������������������������ûŻлۻۻлȻû����������»ûûûûûÿĿѿݿ�������"�����ݿ˿������Ŀm�y�������������y�`�G�;�$��"�.�G�V�`�m�m�y�������������y�m�d�`�^�Y�`�b�m�m�m�m�B�O�[�h�~��t�m�d�[�O�B�<�6�4�0�0�2�6�B���ʾ׾۾�����ܾ׾ʾʾ��������������G�`�y�����ƿ¿����y�T�"�	��߾�	���G�������������������������Z�f�n�s�x�s�r�f�]�Z�M�G�M�P�T�U�Z�Z�Z�Z����������������������������������������������(�5�N�Z�_�V�R�L�A�(���ٿȿ̿ӿݿ�����������������������þ�����������������������������������������������������
��/�4�9�<�B�A�<�9�/�#��
�����������g�s�������������������g�Z�N�I�J�N�Z�\�g�Z�_�f�i�f�f�_�Z�P�M�L�M�N�V�Z�Z�Z�Z�Z�Z����	�G�r�k�G�;�.� �	����ʾ����������������������������������������������������uƎƚƤƳ��������ƳƭƧƚƎƊ�h�W�X�\�u�����ľ��������s�f�Z�4�#�%�4�;�A�M�f����F1F=FJFVFYF`F\FVFNFJF=F1F%F$F FF$F,F1F1�A�N�d�q�s�m�g�_�`�]�N�(��� � ��%�5�A�"�/�@�F�H�C�;�5�/����������������	��"�t�g�[�N�5������)�N�g�t���(�4�M�Z�v�����s�f�M�4��������ìù��������������ùìàÓÜàåìììì�"�/�H�l�z�~�t�a���������������������"����!�%���������ù����������������׾׾�����׾ʾ����������ʾԾ׾׾׾׻-�:�A�F�K�F�<�:�3�-�!� �!�$�-�-�-�-�-�-�����������ƽͽĽ������������y�v�o�{�����#�0�<�A�G�E�<�8�0�,�#��
�	�
����#�#���������������������������������������ػ��������������ûȻû����������x�x�{�|�����
��#�/�:�/�&�#���
�����������������лܻ������������ܻڻлͻлллм�'�4�6�=�@�4�'�!����������������������������������������������������U�a�n�zÇÍ×Ò�z�a�U�H�>�H�K�E�L�Q�N�U�����������������������������������������g�s�������������������s�g�Z�N�M�K�P�Z�g������(�(�1�)�(��������������������zÇÍËÊÏÈÇ�z�y�n�a�U�P�U�Z�a�k�n�z����������ٺֺкֺ׺���������
��#�0�<�I�U�_�U�Q�I�<�0�#�� ������
ŔŠťŠśŔōŇŃ�{�w�n�j�n�{�}ŇŔŔŔ�������������������x�l�_�Y�T�W�_�d�x�����Y�e�r�s�|�r�e�d�Y�T�Y�Y�Y�Y�Y�Y�Y�Y�Y�YD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��L�Y�r�~�����������r�e�Y�L�I�@�=�=�@�G�L�������������պҺɺ���������������������E7ECEPE\EaEfE\EPECEBE7E5E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�E�E�E{ExEuEpEjEuEu���������������������������������������� B 9 J � N ? C G I g 3 P E � Z 4 M O / - E b ` 7 U f # $ < D Q < 1 L = V $ & 1 H < w I @ B @ ' ) T M [ L n q 5 # Z P A �  k  �  F  }  q  �  �  n    �  '  �  o  �  �  �  �  >    d  �  �  %  �  (  C  >  �  �  x  �  �  �    �  6  �  *  �  Z  �  �  �  �  "    �    J  �  
  �  +  <  }  �  v  }  U  9��h��1�e`B�e`B�#�
�t���o<��
<�h%   <���:�o=y�#:�o;�`B<t�=C�<D��=+=+=+<��
=�+<�=49X=�+=q��=u=,1=D��=e`B=e`B=��P=u=,1=#�
=]/=@�=8Q�=�%=m�h=ix�=}�=e`B=��=�E�=��-=���=�1=�C�=��
=�7L=��=���>%�T>%=�
=>J>\)> ĜBi'B.KB�*B�^B"��B(YfB!�B	(B~!B�@B�Bs'B
GB@B��B+�BqB�B�vBn�B ��B�EB�KBBںBy�B��B>lA��/B
��B#Z|B!�BABB3-TB��B,h�B�SB1B)yB�B��B�mB�B�BQB�BB=IBz�B&c�A���A��]B
�Bo�B:B��B@�BUB�gB�BF�B?2B@cBϛB"�GB(=�B!�B?�BB�B��BK�B@�B
O*B��B��B�BP�B9�B��B��B E�B��B�1B=�BAuB?�BR;B}-A��oB
��B#C�B"?�B��B�tB3E�B��B,��B�B:8B*B �B�~B��B�BE}B>hB� B~!BQ�B&Z!A�RTA���B��B@B:cBB��B@{B�pB�{A��2AS�LB
�#A�X(@��@��@���A��Ah�GAlkIA�x�AR�Aj�~?0&aA?g&A���A���A�T�A�$�A���A��A>܃AT��A���B�AB+oC��dA��'A���A���A9R�A��A�M_A��tARe@w��A l�A��DA�w�@���A�`�@�@�o�A�rbA�9�A�iDA�yrA3eA�-)@E��A�wA��@��`?�"cC���?��@!�(C��C�v@�^YA��AS�rB
��A۝�@�&@��@���A��Aj��AlA�{uAR�AAn�?"��AA
ZA�͖A��rA�rRA�z�A��LA��A>��AS�iA�~
B@�AB�$C���A��7A�VSA�wRA7kÀHA��A��5AQ�]@tA�A�aA�kA�L�@�S<A�g@@�Y_@�%�A��.A�͑A���A�A16AȄ1@D�A��A�җ@�F�?�k*C��	?��@$��C��C��@�%>         
                  $             @                              3         .   '   '         "      -                        
   	            '            
         %      N   (      
      '                           #            E            )                  =      +   /      #      '   !      3                                                                                                                        9                                    +         #      '         1                                                                                 NK7�N��#O�N�sN0��N��9N6��N�bbO��AN0�OKQN�<eP��NTN���N�W�O�NLQN�O�	SN���NZZEO��`OYg�P�4OOJ�O��O��2Oe]<P��O�fpN��jPU�O#��N��N�
OE��OēN���O��N�B-Na�]N��N���OA�O�OH�UN�%ONdK�Nd]6OW1�N�jbOR�oN�OP�OTKuO�yNL�}O)VO�	B  C  �  �    l  J  �  �  W  U  �  �  �  �  <  �    �  m  �    �  �  �  �  �  	�    �  l  �  �  X    U  V  2  �  �  �  I  �    �  (    �  t  
E  �  :  1  J  M    h  %  c  &  ��o��/��j��C��T���T���t�;�o;D���D��;ě��D��;��
��o;o;�o<u<o<�t�<e`B<�C�<�o=�P<��
<�1=,1<�j<���<ě�<�j<�`B=\)<��=,1=\)=�P=�w=#�
=#�
=<j=H�9=H�9=H�9=L��=aG�=}�=]/=q��=�hs=q��=u=y�#=��=���=ȴ9=�9X=�-=��=��=�F��������������������mllmnz������znmmmmmm������������������������������������������������������������`]^bnw{���{nmb``````��������������������558=BLNT[^][[RNBBA55���
#*/46:=<2/ 
���������������������!$)-5<BNV[][WNB53)!''')05BNUNLBB;5)''''.+-B[h������������B.^ht{����ttsh^^^^^^^^��������������������!#)/5:<C@</#�������������������������������������///6<HLTOH?<6///////9?ACHUWan������znaH9yz|�������������}zyy�)-)#������	)6EIJEB6mprz�������������zqm����,5BQcgsqg[4��  #/<CJQTRH</(# :86;<HUYaciia^UHC<::PMOUan����������naUP!"$/;HPTUUUTLH;/)%"!Sgt������������xk_]S�������
	������������������������������"*/,)��������������������������hhuu��������yuihhhhh_chntvtrtuttth______�����������������������')*)$!��	$)*,)&)16::<<65*&��������������������@?ABFNO[_][VQODB@@@@XWZ[htxtph[XXXXXXXX���� �����������������������������455<BNW[bilhg[NGB:54������������������������������������������������������������-05<IUTID<10--------:89?HU]ahmttojdTHA;:T]ajmnmlmnmca_UTRTTTiinp�������������{ni����������������������

�������JLOSU`nz�����}zniaUJ'()/37BO[bc][OHB60)'����������������������������

������������68?=2)�������������������������������������������ؾʾ׾��������׾оʾ��þʾʾʾʾʾ��$�0�=�>�I�Q�S�U�J�I�=�;�0�+�%�$��!�$�$�t�w�x�z�t�h�d�a�h�n�t�t�t�t�t�t�t�t�t�t���������������������������������������������������������������������������������ûл׻ػлƻû����������ûûûûûûûÿݿ����	����������ݿտѿѿѿܿݿm�y�������������y�m�`�;�9�.�-�;�A�M�T�m�m�y���������y�m�`�^�`�g�m�m�m�m�m�m�m�m�B�O�[�e�h�k�h�f�a�[�T�O�I�B�;�6�5�6�:�B���ʾ׾۾�����ܾ׾ʾʾ��������������;�G�T�k�������¿������y�h�\�G�����;�������������������������Z�f�n�s�x�s�r�f�]�Z�M�G�M�P�T�U�Z�Z�Z�Z�����������������������������������������ݿ�����(�=�H�E�A�;�(������޿Կڿ�����������������������þ������������������������	�������������������������������
��/�3�7�<�A�@�<�8�/�#��
�����������g�i�s�����������s�g�^�Z�N�N�M�N�Z�e�g�g�Z�_�f�i�f�f�_�Z�P�M�L�M�N�V�Z�Z�Z�Z�Z�Z���ʾ׾�������	���׾ʾǾ��������������������������������������������������uƎƚƤƳ��������ƳƭƧƚƎƊ�h�W�X�\�u�Z�f�s�������������s�f�Z�M�C�C�F�M�T�ZF1F=FJFVFWF^F[FVFMFJF=F1F'F$F F F$F.F1F1�5�A�N�X�f�l�Y�[�X�N�A�(���
����.�5�/�7�;�D�G�B�;�2�/��������������	��"�/�t�g�[�N�5������)�N�g�t��(�4�A�Z�m�r�f�_�Z�M�4���������ìù����������ùìàØßàéìììììì�"�/�H�k�y�m�T�/���������������������"��������� ����������������������׾׾�����׾ʾ����������ʾԾ׾׾׾׻-�:�A�F�K�F�<�:�3�-�!� �!�$�-�-�-�-�-�-�������ý��������������y�u�q�y�}���������#�0�<�A�G�E�<�8�0�,�#��
�	�
����#�#���������������������������������������ػ��������������ûȻû����������x�x�{�|�����
��#�/�:�/�&�#���
�����������������лܻ������������ܻڻлͻлллм�'�4�6�=�@�4�'�!����������������������������������������������������a�g�n�y�zÇÊÓÌ�z�n�a�Y�U�Q�J�Q�V�Y�a�����������������������������������������g�s�����������������w�g�Z�O�N�M�S�Z�c�g������(�(�1�)�(��������������������zÇÈÇÄÂ�z�u�n�j�a�_�a�n�n�y�z�z�z�z����������ٺֺкֺ׺���������
��#�0�<�I�U�_�U�Q�I�<�0�#�� ������
ŔŠťŠśŔōŇŃ�{�w�n�j�n�{�}ŇŔŔŔ�������������������x�l�_�Y�T�W�_�d�x�����Y�e�r�s�|�r�e�d�Y�T�Y�Y�Y�Y�Y�Y�Y�Y�Y�YD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��Y�e�r�~���������r�e�Y�L�J�@�>�=�@�H�M�Y�������������պҺɺ���������������������E7ECEPE\EaEfE\EPECEBE7E5E7E7E7E7E7E7E7E7EuE�E�E�E�E�E�E�E�E�E�E�E�E{ExEuEpEjEuEu���������������������������������������� B 9 J � N ? N A M [ 4 P E � Z 2 R O ( , 4 b ' 7 U F !   ; D D 9 , . = V ( & 1 H < w I @ 2 / % ) a M [ L n q ) " Z P A �  k  �  F  }  q  �  q    (  Y  F  �  �  �  �  �  y  >  �  I    �  B  �  (  �  *  �  �  x  ,  �  �  _  �  6  �  *  �  Z  �  �  �  �  �  M  �    �  �  
  �  +  <    �  v  }  U  9  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  ?m  C  A  >  <  9  7  5  5  8  :  =  ?  B  G  S  `  l  x  �  �  �  �  �  �  �  �  �  �  �  x  d  M  6    �  �  �  �  �  _  �  �  �  �  �  �  �  y  o  c  V  F  3       �  �  n  w  �                    $  +  1  7  :  6  2  /  +  '  #  l  o  s  v  y  }  �    |  z  w  t  r  o  m  j  h  e  c  a  J  I  H  H  G  F  C  @  =  :  4  +  !         �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  Z  E  )  O  g  n  p  v  {  �  �  v  [  ;    �  �  _  �  <  �   �  }  �    7  N  V  L  :    �  �  �  ^  !  �  �  5  �  w  �  D  B  ?  <  <  C  J  Q  V  W  X  Y  [  ^  a  d  e  f  g  h  ,  \  �  �  �  �  �  �  �  �  �  �  U    �  �  4  �  [   �  �  �  �  �  �  �  �  �  z  l  ]  M  >  .       �   �   �   �  O  �  �  �  �  �  �  }  [  0  �  �  �  @  �  k  	  Q  o  �  �  �  {  q  h  ^  T  J  @  6  2  4  5  7  8  :  ;  =  >  @  <  8  3  /  *  $          �  �  �  �  �  |  e  O  :  $  �  �  �  �  �  �  �  �  {  s  j  _  T  H  ;  .  "        �  �  �  �          �  �  �  �  l  3      �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  x  g  W  G  4  !    �  �       9  N  `  k  k  e  U  C  ,    �  �  �  l  @    �    �  �  �  �  �  �  �  �  �  �  k  N  0  �  �  .  �  t     �  q  �      �  �  �  �  �  �  v  J    �  �  �  =    �  �  �  �    w  o  d  T  D  4  $    �  �  �  �  �  d  B      �  �    c  Q  �  �  �  �  �  �  �  �  �  w  L  
  �    b  i  �    t  h  ]  R  E  5  #    �  �  �  �  �  �  �  r  h  ^  �  �  ~  m  X  <    �  �  �      �  �  �  U    �    #    )  B  S  T  M  B  ?  d    w  f  N  (  �  �  (  �    �  	�  	�  	�  	�  	�  	�  	�  	E  �  �  F  �  �  %  �  S  �  O  �  �  �          �  �  �  �  �  u  `  E    �  �  n  	  �    �  �  �  �  �  �  u  \  B  "     �  �    K    �  �  '  �  l  X  D  B  >  7  (    �  �  �  �  ]  ,  �  �  y  2  �  �  �  �  �  �  �  �  }  M    �  �  �  V    �  �  P    l  �  �  �  �  �  �  �  �  �  �  �  T  $  �  �  �  Z    %  �  %  L  U  I  5       �  �  �  I    �  �  \  "  �  �  K  �   �  �  �  �          �  �  �  �  r  ;  �  �  i    �  Q  �  U  O  H  A  8  0  %      �  �  �  �  �  �  �  �  l  S  9  V  L  C  9  /  &            �  �  �  �  �  �  �  �  �    &  0  .  '         �  �  �  �  }  U  '  �  �  �  �  *  �  �  �  �  �  �  �  �  �  x  k  _  V  M  C  9  .      �  �  �  �  �  �  �  �  �  w  m  b  W  O  H  A  ;  I  ]  q  �  �  �  �  �  �  �  �  t  _  B    �  �  y  >    �  �  n  �  I  ?  4  /  +  &             �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  L  %    �  �  �  �  �  �  R  "  �  �  ~        �  �  �  �  �  �  �  �  �  �  �  �  t  e  [  R  J  �  �  �  �  �  �  �  �  �  z  k  \  M  @  7  /  )  3  =  F  �  �  �    '    	  �  �  �  �  ]  .     �  �  ^    �  �  �  �  �  �          �  �  �  n  1  �  �  .  �  �  k    �  �  �  �  �  �  �  {  U  )  �  �  �  ]     �  �  Z    �  t  ^  F  -  %  %     �  �  �  |  R  @    �  �  �  �  s  6  �  	  	)  	J  	l  	�  	�  	�  
  
B  
2  	�  	�  �  �  i  �  �    �  �  �  �  �  �  z  c  K  3    �  �  �  �  �  r  X  -  �   �  :  4  *    �  �  �  �  ~  \  8    �  �  J  �  �  h  Z  �  1       �  �  �  �  �  �  �  �  �  �  �  �  y  n  [  G  4  J  "  �      �  �  �  f  $  �  �  �  "  �  �  d  �  #  a  M  D  ;  1  (          �  �  �  �  �  �  �  �  �  �    E  �          �  �  d  �  n  �    d  �  �  �  
�  	�  4  g  S  O  F  8  '      �  �  �  _  $  �  �  A  �  �  >  �  %           �  �  �  �  �  �  w  K    �  �  p  /  �  S  c  ^  X  F  2    �  �  �  �  g    �  0  �  �  o  C     �  &    
  �  �  �  �  l  7    �  �  X    �  Y  �  7  �  �  �  �  U    
�  
�  
B  
	  	�  	�  	C  �  ^  �  D  �  �  �  $  �