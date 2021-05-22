CDF       
      obs    2   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���`A�7      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�z   max       P�S�      �  t   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �T��   max       >O�      �  <   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E��z�H     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vd(�\     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @P�           d  /�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��          �  0   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       <t�   max       >k�      �  0�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�jI   max       B,�G      �  1�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B,��      �  2`   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >@�    max       C��      �  3(   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >A�D   max       C���      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  6H   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�z   max       Pj��      �  7   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Xy=ـ   max       ?�7KƧ�      �  7�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���
   max       >�+      �  8�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E��z�H     �  9h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vd(�\     �  A8   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @P�           d  I   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�J           �  Il   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  J4   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���%��2   max       ?�4֡a��     �  J�   &      
            4         +   \   	   +                  k         "      +      '   
         r            �         4            	               >      o      PG�N���Nz�M�zN	K!OrņPK�hN�;0NI��P*��P�S�OlpPO�HO]]N�-�Od*"O	�uNæ�P�dDN��?NUY�O���O�dPj��N��"PCKN��SN�qsN�E�PH=�O-�\O7�:NS��Pr	�NڳhN:q�P*a2NLX�O5,�Nx�N�3ZN�BN���NP[�N��@O�N9N�[2OX��O
�uN-�:�T����o%   ;�`B<o<t�<49X<u<�o<�o<�t�<�t�<�t�<�1<�1<�1<�9X<���<���<���<���<�`B<�h=o=\)='�='�=,1=,1=0 �=49X=8Q�=@�=T��=]/=m�h=m�h=u=y�#=�\)=���=��=ȴ9=���=�S�=�x�=�h=�F=��m>O�5Bgtz|�����hB)UP[_htt��������th[UU~xx���������~~~~~~~~����������������������������������������ghu��������������tqg"*6O[t��������h[6"��������������IKNR[fgig`[NIIIIIIII��������#*-@D;/
������05BGadajg[N) �� )57:>=75)$HNO\\m��������t[NIBH����
#*-//-#
��,-/<HIMMH<8/,,,,,,,,}���������������������������������������������������������������)7?SNB.�����wz������������wwwwww����������������������������
�������������������������ZRYg�����
#(������Z006BO[a^[OB60000000035B[�������vtg_YNB63]_hhqt�������}th]]]]����
#,#!
�����A=<?CHUZacda^VUHAAAA��������
��������������dknpvz�����������znd9;;<=HMPUUUUJHB<9999�������!�����������������������������������������������������)+'
��������LOU_bnr{�{nbaULLLLLL%)5=BJMIB76)"///'"��������������������//<>@@<<4/)#" #/////��������������������nla`\X[agnnpponnnnnn��� "#������������"#"
�����"#(&$# ��������	
��������������

����������������������������G�m�������������m�T�;�.�+�,�"����"�G�����������������������������������������@�L�Y�a�d�_�Y�P�L�@�8�;�@�@�@�@�@�@�@�@�T�Y�a�b�h�a�a�a�T�T�Q�T�T�T�T�T�T�T�T�T�)�6�6�6�*�3�)�����(�)�)�)�)�)�)�)�)�����������������������z�s�p�k�i�l�s������������������������e�P�O�K�K�M�e��!�-�:�;�F�L�F�A�:�-�!��������!�!�b�n�u�{ł�{�v�n�k�b�^�Z�b�b�b�b�b�b�b�b��������ʾ�	����	�㾾����������s���������=�P�Q�=���Ƴ�h�N�;�@�\�hƎƩ���f�s�t�|�������s�f�Z�M�I�J�M�R�Z�`�f�f�	��/�;�H�T�\�Z�H��������������������	�m�y�������������y�m�a�^�T�P�S�T�Y�`�g�m�;�G�Q�R�I�G�;�.�&�'�.�5�;�;�;�;�;�;�;�;�����ùϹ����������ܹϹù���������àìùý��������þùìàÓÈËÓÓÝàà�M�Z�f�m�o�h�f�c�Z�S�M�C�A�<�A�D�M�M�M�M�U�nŇŏŇ�n�b�T�I�#��ĴĦĳĿ������0�U�<�H�J�K�H�A�<�0�/�/�%�&�/�2�<�<�<�<�<�<�:�F�S�]�W�S�F�@�:�:�4�6�:�:�:�:�:�:�:�:�������ʼмּڼڼ�ּʼ�������u�t�y�����s���������������g�Z�����&�8�N�Z�g�s���5�N�X�]�V�<�@�>�4�(����������ݿ�������ݿܿؿտؿݿݿݿݿݿݿݿ��O�^�zĀ�w�z�t�h�[�B��������)�3�5�B�O�-�:�F�G�S�X�\�S�J�F�:�:�0�-�)�)�-�-�-�-������
�����������������������������������)�/�/�)�������������������Óàð��������������ùà�z�n�e�`�l�zÌÓ�;�H�T�a�m�z�����z�n�m�a�T�H�D�;�4�4�:�;����	���.�6�7�6�)�%��������������������������������������}�����������������ʼּ����ټ��������r�_�E�M�Y������ʽ����Ľнݽ��ݽнĽ��������������������A�N�Z�]�`�Z�N�A�?�9�A�A�A�A�A�A�A�A�A�A������'�0�2�,��	���������m�g�_�s������������	�	�� ����������������r�~���������������ɺԺ��������~�|�q�q�r�
���#����
���
�
�
�
�
�
�
�
�
�
�������������Ľ�������������������������ǭǭǤǡǔǈ�{�z�{�|ǈǓǔǡǦǭǭǭǭǭ�������������������������x�q�x���������������������ûлֻлûû�����������������¦ª±¨¦�{�~ŇŔŠŢŦšŔŐŇ�{�n�b�U�L�Q�Z�a�k�{Ň���*�6�C�O�Q�O�O�C�6�*�$�������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDuDwD{D�D�EuE�E�E�E�E�E�E�E�E�E�EuEoEnEjEiEuEuEuEuE7ECEPEWETEPECE<E7E1E7E7E7E7E7E7E7E7E7E7 ? N 0 ~ p K ` ( D 8 N 1 B : 7 [ 6 ( g ^ 8 ) c p b U O $ & 0 A ? k G t 5 h y ^ H N B h � 2 @ o ' + C  �  8    ~  ^    �    i  "    +  �  <  �    -  �  	  �  f  m  �  �  �  �  �  �  �  p  z  �  �  P  U  N  �  �  �  4  �  �  �  �    +  �  �  9  Y<���<u<t�<t�<D��<e`B=}�<�`B<�`B=m�h=�;d<���=y�#=8Q�<�/=49X=8Q�=�P>1'=��=+=y�#=P�`=��P=#�
=���=L��=T��=e`B> Ĝ=�7L=y�#=L��>Kƨ=�+=�+=�;d=��=���=���=�1=�"�=�/=�;d>%>333>o>k�>��>��B�B'�B,Bh8B[bB
��B�\Bf�B�8B4�B>�B��B
[B�B��B�AB!�B��BݼB;�B!gB#3�B��B�Bi�B	�YBH�B�+B�HB�yB>�B��BLHB�B"�oBEB5�B'׾B��A�jIB+�B\�B,�GBv�B�+BC�BR�B�	B��BB-B�RBAcB��B�@B�!B�B��BF:B��B��B1�B��B
?�B:�B?�B�B"<	BǚB?�B��B!7BB"�(B�$BG�B��B	gBHeB��B�&B��B?�B-�B@�B�B"��B �B�:B'��B@�A�B*��BA�B,��B?+B�)BǧB3�B�(B<>BBvAfΰ@�ŗ?�F�A���Aֱ�AF��ADn�@p��A�SAQ�B�xA@1�A��/AkmhAcM2>@� A�#<A>��A��TA�-@�@�EOA���A��A}�9A���@~J�A�f�A��`A�:�A��*A�J�A���@�<8A$^KA���A�o�@N"@B�A�~A"(B@	@��&@��A��A�n!B  �C�ӏC��C��bAf�+@�X�?ΈA�sZA�vAD�AD�w@r��A��AOaB-�A?@�A��Ak`�Aa/&>A�DA��mA=��A腔A�@�/+@�gzA�c�A��
A~�A�W@��A��A��VA�i�A���AԀA��@���A$�A�}�A��M@S�x@��A�InA! �BK�@���@��A�mA�n<B t9C�׺C���C���   '      
            4         +   \   	   ,                  l         "      +      '   
         s            �      	   4            
               >      p         -                  /         )   C      3                  ;            )   9      +            -            9         7                                          #                              5      #                  )            )   9      +                        #         /                                       Pe4N���Nz�M�zN	K!OrņOgt%N���NI��O�ZXP^~KN��mO��N��}N�-�N:j�O	�uNæ�P{XN]�oNUY�O]��O�dPj��N��"PCKN��SN�qsN�,O�ܖO-�\O7�:NS��O�srNڳhN:q�P+NLX�O5,�Nx�N�3ZN�BN���NP[�N��@O�N9N�[2OͺO
�uN-�:  N  �  �  �    �    �  �  g    r  �  �  !  *  �  8  
�  t  �  J    �  �  !  �  �  �        G    �  #  �  �    �  <  ;  �      Y  �    	u  軣�
��o%   ;�`B<o<t�=�w<�o<�o<���=0 �<��
=+<�j<�1=C�<�9X<���=u<�/<���=t�<�h=o=\)='�='�=,1=<j=�{=49X=8Q�=@�=�/=]/=m�h=}�=u=y�#=�\)=���=��=ȴ9=���=�S�=�x�=�h>�+=��m>O�5B[gtvtz�t[B)UP[_htt��������th[UU~xx���������~~~~~~~~����������������������������������������ghu��������������tqgZW]ht������������thZ��������������IKNR[fgig`[NIIIIIIII������#(/9<=<80#�����)5NV[YZ][NB&	� )5:95)Y^^cg{����������ga[Y����
#(,,#
�����,-/<HIMMH<8/,,,,,,,,�����������������������������������������������������������������)3=CD;5)��z�����������zzzzzzzz����������������������������
 ��������������������������ZRYg�����
#(������Z006BO[a^[OB60000000035B[�������vtg_YNB63]_hhqt�������}th]]]]����
#,#!
�����CACHJUV_aYUHCCCCCCCC���������  ������������������dknpvz�����������znd9;;<=HMPUUUUJHB<9999��������������������������������������������������������������')&�������LOU_bnr{�{nbaULLLLLL%)5=BJMIB76)"///'"��������������������//<>@@<<4/)#" #/////��������������������nla`\X[agnnpponnnnnn��� "#������������"#"
�����"#(&$# �����������������������

����������������������������G�`�m�y����������y�m�T�G�;�0�%��#�.�G�����������������������������������������@�L�Y�a�d�_�Y�P�L�@�8�;�@�@�@�@�@�@�@�@�T�Y�a�b�h�a�a�a�T�T�Q�T�T�T�T�T�T�T�T�T�)�6�6�6�*�3�)�����(�)�)�)�)�)�)�)�)�����������������������z�s�p�k�i�l�s��s���������������������s�l�f�c�`�`�d�s�!�-�:�:�F�J�F�>�:�-�!������!�!�!�!�b�n�u�{ł�{�v�n�k�b�^�Z�b�b�b�b�b�b�b�b�����ʾ�����������׾�������������������������������ƧƎ�h�_�Z�_�uƎƧ�̾f�k�s�s�s�r�n�f�Z�S�N�U�Z�e�f�f�f�f�f�f������"�*�;�D�@�;�"�	������������������m�y�����������~�y�m�`�T�U�[�`�h�m�m�m�m�;�G�Q�R�I�G�;�.�&�'�.�5�;�;�;�;�;�;�;�;���ùǹϹ۹չϹ̹ù���������������������àìùý��������þùìàÓÈËÓÓÝàà�M�Z�f�m�o�h�f�c�Z�S�M�C�A�<�A�D�M�M�M�M�#�<�I�S�V�S�I�?�0�
������������������#�<�H�H�J�H�@�<�/�&�'�/�4�<�<�<�<�<�<�<�<�:�F�S�]�W�S�F�@�:�:�4�6�:�:�:�:�:�:�:�:�������ɼʼҼԼѼʼ���������}�|���������s���������������g�Z�����&�8�N�Z�g�s���5�N�X�]�V�<�@�>�4�(����������ݿ�������ݿܿؿտؿݿݿݿݿݿݿݿ��O�^�zĀ�w�z�t�h�[�B��������)�3�5�B�O�-�:�F�G�S�X�\�S�J�F�:�:�0�-�)�)�-�-�-�-������
���������������������������������(�)�*�)���������������Óàù��������������ìÇ�z�x�s�s�v�{ÇÓ�;�H�T�a�m�z�����z�n�m�a�T�H�D�;�4�4�:�;����	���.�6�7�6�)�%��������������������������������������}���������������������ʼּ޼��߼ּ�������������~�����������Ľнݽ��ݽнĽ��������������������A�N�Z�]�`�Z�N�A�?�9�A�A�A�A�A�A�A�A�A�A������$�-�/�(��	�����������~�s��������������	�	�� ����������������r�~���������������ɺԺ��������~�|�q�q�r�
���#����
���
�
�
�
�
�
�
�
�
�
�������������Ľ�������������������������ǭǭǤǡǔǈ�{�z�{�|ǈǓǔǡǦǭǭǭǭǭ�������������������������x�q�x���������������������ûлֻлûû�����������������¦ª±¨¦�{�~ŇŔŠŢŦšŔŐŇ�{�n�b�U�L�Q�Z�a�k�{Ň���*�6�C�O�Q�O�O�C�6�*�$�������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D�D�D�D�D�EuE�E�E�E�E�E�E�E�E�E�EuEoEnEjEiEuEuEuEuE7ECEPEWETEPECE<E7E1E7E7E7E7E7E7E7E7E7E7 * N 0 ~ p K M + D ) N I I 3 7 j 6 ( R S 8 ( c p b U O $ ) . A ? k * t 5 ^ y ^ H N B h � 2 @ o  + C  i  8    ~  ^      �  i  �    �       �  s  -  �  �  �  f  �  �  �  �  �  �  �  �  �  z  �  �  �  U  N  A  �  �  4  �  �  �  �    +  �    9  Y  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �    =  N  J  <  +    %      �  �  t    �  #  �   �  �  �    v  n  c  S  B  -    �  �  �  �  �  �  {  D  	  �  �  �  �  �  �  �  �  �  �  v  h  Z  K  :  #    �  �  �  �  �  �  }  y  v  r  n  k  g  c  h  u  �  �  �  �  �  �  �  �    �  �  �  �  �  t  g  [  N  9    �  �  �  �  �  y  b  J  �  �  �  �  �  v  l  c  X  K  ?  3  +  &  !           �  t  �  �  �  �  �  �  �  �  �      �  �  �  "  �    �  �  �  �  �  �  �  �  �  �  �  �  l  M  +    �  �  (  �  �  W  �  �  s  O  '  �  �  �  �  n  6  �  �  n     �  �  /  �  �  j  �    S  f  e  T  =    �  �  l    �  ~     �  �  u  m  a  �  �  �  �      �  �  M  I  @    �  Y  �  �      �  [  U  O  V  f  q  m  i  b  [  S  H  =  -      �  �  �  O  J  v  �  �  �  �  �  �  �  �  �  �  �  f  +  �  �  )  �  D  �  �  �  �  �  �  �  �  x  J    �  �  2  �  �  1  �  �  �  !              �  �  �  �  �  �  �  �  �  �  �  1  x  %  |  �  �  o  �  Y  �  �    !    �  �  f  �  F  �  =  �  �  �  �  �  �  r  S  /    �  �  h  '  �  �  l  +  �    7  8  4  -  $    	  �  �  �  �  �  �  �  }  d  7  �  �  U    	  	�  	�  
o  
�  
�  
�  
�  
�  
�  
�  
M  
   	o  �      �  �  �  e  n  r  m  c  W  E  -    �  �  �  �  x  O  &  �  �  �  �  �  �  �  �  �  s  c  R  A  .      �  �  �  �  �  
    #  �  �    7  H  I  @  /    �  �  �    *  �  o  ^  -  �  �    h  U  >  !  �  �  �  �  Z  *  �  �  �  �  `  '  �  �  ,  �  �  �  �  �  �  �  �  �  |  a  E  %  �  �  |    �     �  �  }  n  `  Q  B  2  "      �  �  �  �  �  �  �  z  b  J  !    �    �  �  �  q  @  2    �  �  c    �  �  -  ~  A  �  �  �  �  �  �  �  �  ~  g  O  6    �  �  �  �  Z  9    �  �  �  |  r  b  Q  >  *    �  �  �  �  h  7  �  �  �  N  �  �  �  �  �  �  �  �  �  �    d  D    �  �  �  R    �  
�  m  �  `  �  �        �  �  [  �  D  
j  	|  x  7  %   �    �  �  �  �  �  �  �  m  S  8    �  �  {  1  �  �  *  J      �  �  �  �  r  V  E  J  S  M  1    �  �  ]    �  1  G  <  2  '        �  �  �  �  �  �  �  �  �  �  �  �  �  F  G  �  L  �  �  �      �  r  �  T  �  
�  	�  �  �  �  �  �  �  i  O  0    �  �  �  T     �  �  �  [  =      �   �   �  #      �  �  �  �  �  �  q  Q  0    �  �  �  �  d  @    X  x    n  p  P  )  �  �  �  H    �  [    y  �  ;  �  �  �  �  �  �  �  k  ^  ?    �  �  �  D  �  �  a    �  h      �  �  �  i  :    �  �  �  �  d  6  
  �  �  X    �  �  �  �  �  �  �  w  g  V  E  1    	  �  �  �  �  �  S  %  �  <  8  3  -  '      	  �  �  �  �  �  �  g  7  �  �  ?   �  ;      �  �  �  �  m  I  &    �  �  �  v  M  "  
  �  �  �  �  n  N  ,  �  �  �  �  _  K  D  4      �  �  �  _  "    �  �  �  �  �  �  �  Z    �  n    �  |  *  �  �  L      n  Y  B  '    �  �  �  �  y  S  )  �  �  ?  �    �  C  Y    �  �  F    �    2  
�  
�  
C  	�  	L  �  �    0  :    �  h  ?  !  
  �  �  �  �  f  B    �  �  �  x  M     �  �  ~  �  �  �  �    �  �  =  �  �    j  �  �  �  �  	�  k  1  	u  	G  	  �  �  �  L    �  x  !  �  R  �  v  �  x  �  �  �  �  �  �  M    �  �  �  g  (  �  o    �  F  �  v    �  :