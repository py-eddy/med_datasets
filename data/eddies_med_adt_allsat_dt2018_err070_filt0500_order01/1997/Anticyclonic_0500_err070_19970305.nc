CDF       
      obs    1   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�E����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�t2   max       Pwr�      �  p   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��9X   max       =��      �  4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @E.z�G�     �  �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @ve�����     �  '�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O�           d  /H   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�b�          �  /�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >y�#      �  0p   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B(�N      �  14   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�y   max       B(�~      �  1�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?j]�   max       C��      �  2�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?o p   max       C��      �  3�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  4D   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  5   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  5�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�t2   max       P&��      �  6�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�l�!-w   max       ?�_o���      �  7T   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       >O�      �  8   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�G�   max       @E.z�G�     �  8�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(�    max       @ve�����     �  @�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O�           d  H,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @�e�          �  H�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CN   max         CN      �  IT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_ح�   max       ?�X�e+�     P  J            5         X            1      K                     Z                            P                              >      
          
   �   =         ,O~�N�� Nm�[P9��N@J�N���P.�,Nu��N?*.Of�MPffN��>Pwr�O(=�NϺ�NQ��N�uiOq.O_�O�PN�QOg6N�yN��/N�%N�]?O�C�OQ�PUO�N��ND��N<�N�0�OW��N�jM�t2O_CP"��N�.N�-Ov-�N�c�N�&ZPEs�O�{O��N�s�O�<���9X�T���T���49X�t��D��%   ;o;ě�;�`B<#�
<D��<T��<T��<e`B<u<�t�<�1<�j<ě�<�/<�<�=+=C�=\)=\)=\)=t�=t�=t�=t�=t�=�P=�P=�w=<j=@�=@�=D��=]/=�%=��=�O�=�\)=�\)=��-=��=��wu{���������������wwA@BB@@BOPS[\bec[OBAABBBBIOZ[]_][TQOBBBBB����
#<GNRH</#�����/++6=BHLB6//////////hjtt���������thhhhhhEDHTh���������th[WPENNY[gjtutqg[PNNNNNNN�������������������������������������������
#<UanyrUF:4�����������������������)Nt������tNB5
�##)/<CHUU`aiaUH</'&# #&/8<CCBB?</#    ����������������������������������������EEO[h��ywx|}}th[XTOE4-**.5BN[`_\Zglk[NB4�������
	�������
 )+-+)





�������������������������������������������������"&)+57:9750+)#(()*05@BIJBJLEB52)((���
#/<BDLHB<3#
�)069;:;61)������������������������
#+/10/.#
�������������������������������������������yvrqv{�������{yyyyyy������������������������������������������� 

#"
�������spt������tssssssssss�����������������������������
�������������������������)32.)#�������������������� ����������������������������������)6OXdmx|t[VOB>8)"aaedcfnz������{znaaaaamz}�����zmmaa`aaaa������� ##����ĚĦĳķĿ����ĿĿĳĭĦĚęđđēĘĚĚ�y�����������½��������������y�r�m�p�y�y�нݽ�������������ݽнʽȽнннпT�c�n���������y�`�;�/�(�����.�;�G�T�zÇÓØÓÎÇ�z�w�q�z�z�z�z�z�z�z�z�z�z�L�Y�e�f�n�o�m�e�Y�S�L�H�D�E�L�L�L�L�L�L�������ʾܾ����Ծ�����f�M�7�3�B�M�f���������������������������������������������������������������������������������Ľݽ�� �
������ݽĽ��������z�������ľ������Ǿ;ž¾��������s�^�M�4��%�A�s�������������������������������������������"�;�G�K�L�Z�?�8�/�"�	���������������	�"������������������������������D�D�EEEEEE#EEED�D�D�D�D�D�D�D�D��"�.�0�.�*�"��	��	������������������������������������������������һ��������������������x�_�S�G�G�X�_�x�����G�T�`�m�y�������}�y�m�`�T�M�G�;�3�4�:�GD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������ĿпǿĿ���������������������������������"�.�3�H�R�G�;�.�"�	����оо�������
���������ܻٻܻݻ������������������z�s�f�c�Z�X�Z�\�f�s����O�h�l�uƁƃƁ�v�u�h�\�O�J�C�6�1�6�C�E�O����$�0�6�=�B�=�0�$�����������
���B�N�[�}�}�w�s�r�g�[�N�>�5�2�)���)�5�B�Y�g�r�~�������������������~�m�e�]�Y�O�Yù�������������������à�z�h�i�l�z�àù���(�(�/�,�+�(�"���	����������<�H�M�T�H�H�<�5�/�,�#�#�#�$�/�1�<�<�<�<ÓàìñìãàÓÉÇÓÓÓÓÓÓÓÓÓÓ�����Ľнݽ�ݽнĽ����������������������(�4�A�M�V�T�M�B�A�4�3�(�����(�(�(�(������'�3�8�4�3�'���������������������ʾ˾ʾþ������������������������)�6�@�9�6�*�)�)�&�)�)�)�)�)�)�)�)�)�)�)��(�5�A�H�N�V�_�e�e�Z�N�A�5�(�������������#�.�1�1�'��	���������z�s�����˼������ʼּؼ�ּռʼ��������������������n�{ŇŒŔŞŔŇ�{�t�n�h�n�n�n�n�n�n�n�n�������������������������������~�~�����!�(�-�:�F�S�_�S�F�D�:�-�!���
���!�!EuE�E�E�E�E�E�E�E�E�E�E|EuEpEuEuEuEuEuEu�����ʼ����������ʼ������������������r�������n�f�Y�M�@�4�'���������I�Y�r�����������ɺպպκɺ���������������������������������ŹųŭŨŭŭŹż����������¦²¿������������¿²¦ ? i j  B T : . c t > _ 9 > 9 U / a =   E o M . Y R   P ? E ? >  C  * ` A Q 9 ] I p V 1 Y 5 4 *    _  .  �    H      �  A  H  �  �  4  �  �  U  �  1  �    �  (  �  �  0    [  �  �  D  �  S  �  �  �  �  +  �  W    �     �  �  Z  4  <  �  =��o:�o�t�=,1��o<�o=�9X<t�<49X<�`B=m�h<�o=�9X=�w=\)<�9X<�h=H�9=t�=�l�=o=,1=T��=��=,1=#�
=�+=P�`=�=L��=T��=L��='�=e`B=}�=49X=D��=��-=�;d=�%=�o=��=���=���>y�#>$�=���=\>z�B�BƞB��B��B�BB�B6gB��BtHB!W�B[
B��BM�B�BX�B�B��BڛB3B��B8cB��B �BB�Bq�BoeB�B�B) B�&B"s�B(�NB�B!5�B$k�B
2lB;�BF�B#1qBZ�B��B�-B�B�1BR�B#A��BZB@ZB�4B��B?�B��B��B?�B�B?B!9�B�]B��B?'B�B?gB?gB�*B��B¼B�GB9[B:uB ��B?KB�BBQCBA�B?�B�BBC�B�B"@�B(�~BAsB!>CB$��B
WB>�B?�B#5�B@PB;=B��BCB=!B6BC5A�yBǝA��[A>@A,l�Ah�A�V�?�B�AFэA�6B��A*��AE�PA�}/A��A�k�C�S�A^M�A��m@�g�Ah4�C���Av<�A\�@�B�AC#B�XB	?�A�qj@�A��DA�:A�M�A�HA'�A9��?j]�AL�.A�)JA�&�A��@��TA���A��@w��C��@��@��@&0[A�>�A�6�A��A/�A,�Aj�[A��3?���AF�OA�p�B��A, �AE1A�s�A�&A�u�C�I�A]�AА�@�SAi�C��kAu�oA^©@��LAC�B�YB�A�m�@��A���A���A��A�eSA(�oA9Ck?o pAL҅A�A�A��;A�'@���A�mA�f�@l�C��@���@���@+��A��aA��            6         Y            1      L                     Z               	             P                              ?                   �   >         -            +         -            7      3                                                '                              /                  -   '                                          -                                                                                    +                     %         N�7�N�INm�[OuCN@J�N��OzZ�Nu��N?*.N�*P&��N��>O��N��BNϺ�NQ��NKPN���O_�N�ˮN�QOg6N�x�N��/N��iN�ӾO1� O8�O�hO�N��ND��N<�N�0�OW��N�jM�t2O@��PC�N�.N�-Ov-�N�c�N�&ZO��O�MN��}N�s�OU��  c  �  �  �  �  �  �    j  �  �  �  {  �  @    �  |  �  �  %  �  r  �  t    �  �  	�  �  9  �  Q  �  �  �  �  8  4  �  Z  w  .  �  �  �  <  �  ���t��#�
�T��<u�t�;��
='�;o;ě�<T��<��
<D��=<j<��
<e`B<u<��
=+<�j=��<�/<�<��=+=\)=t�=8Q�=t�=�%=t�=t�=t�=t�=�P=�P=�w=<j=L��=L��=D��=]/=�%=��=�O�>O�=�t�=��w=��=�G�|���������������||||BBCFGIOPW[_b^[OBBBBBBBBBIOZ[]_][TQOBBBBB��
#/4<???4/# �/++6=BHLB6//////////tpt�������tttttttttYWW[bht��������tha\YNNY[gjtutqg[PNNNNNNN�������������������������������������������
#<UanorgU@/#
���������������������$)5BN[mt{��yt[NB,$,++/<HNUYUTHB<8/,,,, #&/8<CCBB?</#    ����������������������������������������_]\_hrtwwvth________4-**.5BN[`_\Zglk[NB4������

���������
 )+-+)





�������������������������������������������������$())358765.)!5**-35BEB@BFJCB55555	 
#./585/,#
	)-68;9:61)������������������������
#+/10/.#
�������������������������������������������yvrqv{�������{yyyyyy������������������������������������������� 

#"
�������spt������tssssssssss������������������������������������������������������)32.)#�������������������� ����������������������������� ��������#!)6OWclwzs[UOB?9)#gfedgnz������zvnggggaamz}�����zmmaa`aaaa����������ĦĳĳľĿ��ĿļĳĦĚĔĔĖĚĠĦĦĦĦ�y�{���������������������y�v�o�y�y�y�y�y�нݽ�������������ݽнʽȽнннпm�y���������~�y�h�`�T�G�;�7�3�8�G�T�`�m�zÇÓØÓÎÇ�z�w�q�z�z�z�z�z�z�z�z�z�z�L�Y�^�e�g�e�d�Y�T�L�J�L�L�L�L�L�L�L�L�L�������������������������s�`�V�Y�f�s��������������������������������������������������������������������������������ݽ�������������ݽнĽ��������Ľнݾ��������ž������������s�e�M�3�0�A�f��������������������������������������������	��"�.�4�9�;�8�/�)�"��	�������������	������
�����������������������������D�D�EEEEEE#EEED�D�D�D�D�D�D�D�D��"�.�0�.�*�"��	��	������������������������������������������������һ_�l�x���������x�l�_�Y�S�_�_�_�_�_�_�_�_�G�T�`�m�y�������}�y�m�`�T�M�G�;�3�4�:�GD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������ĿпǿĿ���������������������������������"�.�3�H�R�G�;�.�"�	����оо�������
���������ܻڻܻݻ������������������z�s�f�c�Z�X�Z�\�f�s����O�Z�\�h�uƁƂƁ�u�r�h�\�O�M�C�7�C�L�O�O�����$�0�3�7�0�*�$������������������5�B�N�[�g�m�r�m�g�e�[�N�B�:�5�)�(�)�4�5�Y�e�r�~�������������������~�o�e�^�\�Y�Yù��������������������ìàÐÇËÔàìù���(�(�/�,�+�(�"���	����������<�H�M�T�H�H�<�5�/�,�#�#�#�$�/�1�<�<�<�<ÓàìñìãàÓÉÇÓÓÓÓÓÓÓÓÓÓ�����Ľнݽ�ݽнĽ����������������������(�4�A�M�V�T�M�B�A�4�3�(�����(�(�(�(������'�3�8�4�3�'���������������������ʾ˾ʾþ������������������������)�6�@�9�6�*�)�)�&�)�)�)�)�)�)�)�)�)�)�)�(�5�A�D�N�T�\�a�_�Z�N�A�5�(�!�����(���������� �,�0�/�%��	�����������������������ʼּؼ�ּռʼ��������������������n�{ŇŒŔŞŔŇ�{�t�n�h�n�n�n�n�n�n�n�n�������������������������������~�~�����!�(�-�:�F�S�_�S�F�D�:�-�!���
���!�!EuE�E�E�E�E�E�E�E�E�E�E|EuEpEuEuEuEuEuEu���ʼּ�������ּ��������������������Y�r�����m�f�Y�M�@�4�'���������L�Y���������ɺӺԺκɺ�����������������������������������ŹųŭŨŭŭŹż����������¦²¿����������������¿²¦¥¦ 5 N j 0 B J . . c F ? _ 2 % 9 U . C =  E o E . [ S  G 4 E ? >  C  * ` A C 9 ] I p V + W . 4 %      �  �  �  H  ?  �  �  A  6    �  �  �  �  U  h  �  �  �  �  (  �  �  �  �  r  �  P  D  �  S  �  �  �  �  +  �  �    �     �  �  q    !  �  �  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  CN  C  V  ^  b  b  [  K  4       �  �  w  :  �  �  R  �  y    �  �  �  �  �  �  �  �  �  �  �  �  m  X  B  )  �  �  ^  )  �  �  �  �  �  �  �  �  �  �  �  �  �  q  b  R  A  1       (  F  s  �  �  �  �  �  �  �  �  �  E  �  �  `    �    �  �  �  �  �  �  �  �  �  |  v  p  k  `  R  D  6  *          @  a  �  �  �  �  �  �  �  �  �  o  0  �  �  H  �  �  9  �  �  +  �  �    :  `  |  �  x  D  �  �  &  �    �  A   �    	  �  �  �  �  �  �  �  �  s  ]  G  0    	  �  �  �  �  j  [  M  >  /      �  �  �  �  �  z  P  &  �  �  �  Z  #  �  �  �  �  �  �  �  �  �  �  �  �  k  ?    �  {    �  9  i  �  �  �  �  �  �  W    �  �  �  |  e  C  
  �  �  h  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  b  O  �  �  -  z  �  �  "  S  w  x  f  J  #  �  �    d  �  k  �  	  E  �  �  �  �  �  �  �  �  �  Z  /  �  �  �  A  �  �  �  @  >  :  2  )        �  �  �  ]    �  B  �  a  �  R  �        �  �  �  �  �  �  �  w  c  W  K  :  "  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  d  W  O  M  @  0      �    *  .  +     f  u  h  K    �  5  �    C  t   �  �  �  �  x  g  Y  J  8  "    �  �  �  e  X  B    �  M   �  e    �    �  �  ;  k  �  �  y  @  �  Z  �  �  
�  	n  �  L  %      	    �  �  �  �  �  �  �  �  �  r  ^  K  9  '    �  �  �  �  �  �  �  �    c  J  2      �  �  �  �  �  �  1  q  g  Y  I  ;  :  :  3  &    �  �  �  �  o  6  �  e  �  �  �  �  �  �  �  �  �  �  �  �  z  g  P  9  !     �   �   �  n  q  s  n  d  Y  J  <  +    	  �  �  �  �  �  �  k  G  #                  
  �  �  �  �  �  �  �  ~  h  S  =  S  e  v  �  �  �  �  �  �  �  �  ~  T    �  }    �  �   �  �  �  �  �  �  �  �  m  ~  l  E    �  �  O  +  	  �  �  �  �  	Z  	�  	�  	�  	�  	�  	�  	�  	�  	^  	  �  F  �  O  �  �  5  l  �  �  �  �  �  l  P  2    �  �  �  j  @    �  �  �    &  9  2  '      �  �  �  �  �  b  @    �  �  �  �  ~     �  �  �  �  �  f  G  (    �  �  �  U    �  �  m  &  �  �  C  Q  F  <  2  %    �  �  �  �  �  �  n  X  B  ,     �   �   �  �  �  �  �  �  }  ]  >    
  �  �  �  �  �  �  �  g  G  	  �  r  U  4    �  �  �  �  q  H    �  �  P    �  �  9  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  g  T  A  �  �  �  �  �  }  z  v  r  n  k  g  c  ^  Z  V  R  N  J  F  -  4  8  5  -    �  �  �  |  J    �  �  V    �  �  ,  [  �  4  /      �  �  �  R    �  �  �  O    �    �  �  0  �  �  �  �  �  m  M  /              �  �  �  �  �    Z  A  (    �  �  �  �  x  X  4    �  �  �  _  �  ~  +  �  w  i  h  ^  W  P  X  ]  _  S  5    �  �  ]    �  h  �  y  .    �  �  �  �  W  /    �  �  q  :  �  �  �  �  l  L  )  �  �  �  �  �  b  D  $    �  �  �  |  \  Z  g  _  S  7    �  E  �    G  o  �  �  �  �  O    �     `  :  �  
  Q  �  �  �  �  �  �  �    Z  N  ,  �  �  �  {  Z    �  N  �  �  *  ;  +    �  �  �  }  ]  ]  G    �  �  ]  �  Z  �    [  �  P  ,      �  �  �  �  �  s  W  :    �  �  �  �  �  �  g  �  �  �  �  �  �  �  q  I    �  v    �  �  :  !  �   �