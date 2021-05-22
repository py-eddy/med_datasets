CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�S����      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�&�   max       P�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       >n�      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F��z�H     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vo
=p��     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O@           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @��           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >dZ      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�F   max       B4�/      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�T�   max       B4I6      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��<   max       C�Xy      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�Z-      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�&�   max       Pl�{      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�{���m]      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >n�      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @F��z�H     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?񙙙���   max       @vo
=p��     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @L�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�         max       @���          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?�z�G�{     �  N�                           
   
   �   )   y            ?   P         7                  !      %   
         �            -   
      o      �         +      N                  +   N]TO���NW%�O��N�W�O���N�e�NV�DN�~@Na>Pb�Pb�P�N�NN�k�O��O���P�OFYNGi�P��N/�NI�VM�&�N�uN'�OO�w�N�g�P"��N�#�Of N��P�O͈{N$J<O�tO$G8N]�N���P7�}N�̯O��NJ�`O
��Oы�NO��O���NO��O�3NL�dN4(�O��OF��NVq��ě��o��o;�o;�`B<o<t�<D��<e`B<e`B<u<�C�<�C�<�C�<�t�<�1<ě�<ě�<���<���<�`B<�h=C�=\)=\)=\)=\)=0 �=H�9=L��=P�`=Y�=ix�=u=}�=}�=�o=�7L=�O�=�hs=�t�=��=��=��=���=���=���=���=��
=��
=�{=�E�=���>n�!#%///4;<=</#!!!!!!��������������������� ��������"/;ADB>=;4/"�������������JILUXYanz������znaUJuuz��������������zuu30<IUYUUI<3333333333��)05-)��������������������������~�������
��������~��������������������������)-08<:,����#/5<DHKHF</*%#dehhkot}�������thdd���������������������������������������������������������������(6:IPQOB6) ��))6BIHB:60-)))))))))347B[ht�������t[OB73������������������������������������������

������������������������������������������������������������������������

�����������)5@IG81)��"(/;FHMIH?;/+"��������

������)+,-) 4117BNg�������tg[NB4�)6DOSUOB6)||��������||||||||||������������������������������������������������������������������������������������������ ���������� !������������

����������

	����������	

#/0573/*#
		�����)6BGI@9&�)))))))))))))����������	
������#08:40#!�� )5:;95/)
'%%)58:9650)''''''''!)/5ABCB<51/+)!!!!!!spnoorvz����������zs���������

	����������������������������(�3�5�A�A�A�5�(�$������������(�C�Z�^�f�j�l�f�Z�N�A�8�(�&��
����A�N�Z�[�^�[�Z�N�I�A�9�<�A�A�A�A�A�A�A�A���	��"�/�;�F�?�>�;�/�"��	�����������������������ùìçåìïùÿ��������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������f�h�n�r�o�m�f�Z�M�H�H�M�N�V�Z�e�f�f�f�f�o�{ǈǔǜǔǓǈ�{�o�j�e�o�o�o�o�o�o�o�o������A�f�s�p�Z���ͽ����۽۽νϽݽ�čĦıĲļĳĚč�t�h�O�B�?�K�[�a�j�tāč��0�I�b�|ŇŔŗŌ�{�b�0������ĿĳĶ����������������������������������!�-�-�:�F�P�S�T�S�F�:�-�!�������T�a�m�q�|�������z�m�a�T�H�;�4�/�-�7�;�T���*�8�6�1�*����������������������������(�N�g�x������s�Z�A�5�������������������������������z�m�l�q�r�z������ÇÉÒÎÇ�z�u�n�g�n�zÅÇÇÇÇÇÇÇÇ�M�Y�f�w�����{�Y�@�4����������	�'�M���������߾�����������������������s�t���������}�s�k�i�s�s�s�s�s�s�s�s�s����������������������������������������ÇÈÓàããàÓÇÇÅÅÇÇÇÇÇÇÇÇ���&�'�3�'���	������������r���������������������f�Y�K�A�M�Y�`�r�;�H�Q�R�M�H�;�8�/�"�����"�&�/�1�;�;������� �9�B�?�2����ѿ��������ѿ����#�*�0�5�7�0�,�#���
���
�����g�t�o�h�^�N�B�5�'���5�B�N�[�g�-�:�B�=�:�-�!��!�$�-�-�-�-�-�-�-�-�-�-���)�6�C�J�Q�Q�O�B�6������������������ʾ���	��	�����о����������������������������������������������������������������������������������y�q�h�g�l�n�v��àìùþýùðìàÜÓÇ��z�y�z�ÇÓà�6�?�B�H�G�B�@�6�)�)�)�1�6�6�6�6�6�6�6�6�������������	�	�
�	���������������������nÇàÿ��������ùîàÇ�z�n�_�e�_�_�a�n��������!�����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDxD{D�D�D����"�.�3�9�.�'�"��	��	��������G�T�`�i�m�y�z�}�z�y�m�`�T�N�G�@�>�>�G�G�~���������Ⱥͺ˺ѺҺκ������~�e�N�O�a�~�@�5�3�'�!��'�3�;�@�D�B�@�@�@�@�@�@�@�@������7�C�F�@�:�-�������ںպغ������� � ���������������������������� ����������ƻƧƒƞƧƴ�����\�h�uƀ�~�u�h�\�O�K�O�V�\�\�\�\�\�\�\�\�	��	���������������������	�	�	�	�	�	�����ʼּ����������ּʼɼ���������EuE�E�E�E�E�E�E�E�E�E�E�EvEuEoEkElEiEmEu�����������ܻлɻλлܻ޻������ F A M A R ) 9 M e W Q E F k O 7 6 H z 4 , ] V k \ P : \ @ ? } b ' M 0 8 8 ] c ; = , \ ( B B  X E j p Y @ d  x  �  �  m  �  �  �  �  ?  �  "  x  �    	  �  �  �  =  c  �  g  �  4  S  G  T  "  �    9     �    3  U  _  W    _  �  �  s  3  �  p  �  �  H  m  �  g  �  ��o<�`B;�o<#�
<�j=+<�/<�t�<ě�<ě�>�R=ix�>O�=�P<�`B=0 �=� �=��=�P=\)=���=C�=t�='�='�=#�
=�7L=P�`=� �=q��=�7L=m�h>Q�=�1=�O�=� �=�/=��-=���>:^5=�E�>dZ=��
=��=��=�1> Ĝ=�{=��=�-=�9X=��>&�y>��B9�B�sB3A�FB�`BNB/B&��B>B��B"��B�/B��Bn BY�B�B�B�B7BՏB�0B4�/B�B��B��B"�B"�Bj�BD*A���B%@B�B	6�BZ�B�B+��B"R;B�B��B°B��B�BQoB��B��B��B"o3B%M�B)vB&�B�Bk�B��BA$B?�BB�B@XA�T�B<`B=�B9�B&�	B9B��B"��BͼB@zB�/B~B��BB�BI�B=	BþB3�B4I6B�VB�hB2B!�B"�&B��B�A�vlBIBΔB	=zB��BNB+�B"A�B��B��B��B�>B9�B?�B�IB��B��B"?�B%��BCpB?B@�B�B>�B)=A��ZA�DPA�[�A�0�A���A�iWC�Xy@��A?��Br9A2P`A�\{A��sA��H@u�A��kA�$�A��A���A���@�\AW)AD9�A��vA�]g@��@�_A���A�')A� �A���@tt:A�(�AQ�A��A;<A��A���A�VA�h�A��/C���A^p]Ah�@{�?��<@^B@M1B�}B4A��A ��C��]@�A"A��-A���A�~A���A��A���C�Z-@���A?~BD�A1�A���A�/Aҁ�@~��A��@A��8A��BA�΂A��A@�ƙAWUAC�rA��oAʉ�@êV@�)PA��A�s�A��A��i@{KA�]AP��A���A�GAʬ�A׃�A��fAʁFA���C��A^��AiN@ѽ?��@[�|@L�^B�B61A�I]A#�C� @��                                 �   )   z            ?   P         8                  !   	   &   
         �            -         p      �         +      O                  +         !                           =   '   7               +         +                        -      !      '   %                  ,               !      !                                                         '   3               %         %                        '                                 #               !                           N]TO&�N��O��N6qOS�"N�e�NV�DN��wNa>O�a�PVPl�{N��LN�k�O}��O�QmO��N���NGi�P�fN/�NI�VM�&�N�uN'�OO�N�g�P��N�#�N���N��O��aO�N$J<O�tO9~N]�N���O��lN`EgO��NJ�`O�uO�BN!m�O���NO��O���NL�dN4(�O��O:�NVq�  �  i  >  �    1  B    M  �  �  �  R  �  0  3  
�  	�  �  �  �  6  �  �  O  Z  �  r  �  �  p  �  �  ^  c  1  �  �  G  E  �    �  �  �  /  
T  �  	    �    
`  ��ě�<49X%@  ;�o<D��<�o<t�<D��<u<e`B=��P<�t�=\)<�t�<�t�<ě�=\)=t�<�/<���=+<�h=C�=\)=\)=\)=�P=0 �=P�`=L��=e`B=Y�=���=}�=}�=}�=�\)=�7L=�O�=��`=���>hs=��=��P=��w=��
=�v�=���=��T=��
=�{=�E�=��#>n�!#%///4;<=</#!!!!!!��������������������"/;ADB>=;4/"��� ���������WX^__anqz�������znaWuuz��������������zuu30<IUYUUI<3333333333 �).3,)      �����������������������������������������������������������������%0473/�����#(/3<DHKHE</+&#dehhkot}�������thdd������������������������������������������������������������)6BFMNB6)$))6BIHB:60-)))))))));67:BO[ht~������tOF;������������������������������������������

������������������������������������������������������������� �����������

����������)5?GF64-)��"(/;FHMIH?;/+"�����

��������)+,-) >=>DNWgt������tg[NB>	)6CORTOB6)||��������||||||||||����������������������������������������������������������������������������������������������������� 	���������������


���������

	����������
#/0573/)#


����)6BFG?8%�))'%)))))))))��������������������#08:40#!��)5:;95.)�'%%)58:9650)''''''''!)/5ABCB<51/+)!!!!!!spnoorvz����������zs�������

������������������������������(�3�5�A�A�A�5�(�$������������(�5�A�N�Z�[�_�_�Z�T�N�A�5�.�(�%����A�N�Q�Z�\�Z�S�N�L�A�<�A�A�A�A�A�A�A�A�A���	��"�/�;�F�?�>�;�/�"��	�����������ù����������ùöìêìöùùùùùùùù����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E������������������������������������������Z�f�k�q�m�l�f�Z�M�J�I�M�O�W�Z�Z�Z�Z�Z�Z�o�{ǈǔǜǔǓǈ�{�o�j�e�o�o�o�o�o�o�o�o�������(�4�>�D�>�(������ؽнսݽ�čĦİıĻĳĦĚč�t�h�O�C�@�K�\�a�tāč�0�I�b�p�|ŋŉ�{�<�0������������������0�������������������������������!�-�-�:�F�P�S�T�S�F�:�-�!�������T�a�m�w�|�|�|�z�v�m�a�T�H�;�7�4�2�;�H�T��������*�1�0�)����������������������N�Z�l�s�|����t�Z�N�A�5��������������������������z�n�s�u�z�������������ÇÉÒÎÇ�z�u�n�g�n�zÅÇÇÇÇÇÇÇÇ�'�M�Y�f�r�}����v�Y�@�4��������'���������߾�����������������������s�t���������}�s�k�i�s�s�s�s�s�s�s�s�s����������������������������������������ÇÈÓàããàÓÇÇÅÅÇÇÇÇÇÇÇÇ���&�'�3�'���	������������r�����������������������f�Y�P�M�Y�b�r�;�H�Q�R�M�H�;�8�/�"�����"�&�/�1�;�;�������7�@�=�0����ѿĿ��������ѿ����#�*�0�5�7�0�,�#���
���
�����[�g�g�g�b�[�W�N�B�5�)�"�&�)�5�<�B�N�T�[�-�:�B�=�:�-�!��!�$�-�-�-�-�-�-�-�-�-�-���)�6�>�B�G�G�B�6�)�������������������ʾ����������;����������������������������������������������������������������������������������y�q�h�g�l�n�v��ÇÓàìùûûùíìàÖÓÇÅ�~�|ÂÇÇ�6�?�B�H�G�B�@�6�)�)�)�1�6�6�6�6�6�6�6�6�������������	�	�
�	��������������������Óàñù����ýìàÓÇ�z�s�m�k�n�u�zÇÓ���������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����"�.�3�9�.�'�"��	��	��������`�g�m�y�z�|�z�y�m�`�T�N�G�@�?�G�T�V�`�`���������ƺ̺ɺϺкʺ������~�e�S�S�Y�i���@�?�3�'�#��'�3�8�@�A�@�@�@�@�@�@�@�@�@������,�;�>�:�5�-�!�����ߺں޺�������� � �������������������������� ����������ƾƧƘƣƧƶ�������\�h�uƀ�~�u�h�\�O�K�O�V�\�\�\�\�\�\�\�\�	��	���������������������	�	�	�	�	�	�����ʼּ����������ּʼɼ���������E�E�E�E�E�E�E�E�E�E�EwEuEoElElEkEuE�E�E������������ܻлɻλлܻ޻������ F D 2 A = # 9 M R W - D F 9 O $ 6 ; r 4 , ] V k \ P 3 \ B ? ] b  B 0 8 . ] c .   \ " ? U   X A j p Y > d  x  w  1  m  Y  �  �  �  �  �  4  |    �  	  �  7  L  v  c  |  g  �  4  S  G     "  �    8     �  �  3  U  $  W    �  g  H  s    �  X    �  .  m  �  g  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  x  q  m  j  g  c  k  w  �  �  �  �  �  �      7  L  ^  g  d  S  7  
  �  �  W    �    _  1  4  8  ;  <  3  )         �  �  �  �  �  �    k  V  A  �  �  �  �  v  k  _  M  7  !  
  �  �  �  �  �  c  3     �  
                  �  �  �  �  �    j  �  �  �    �  �  �  	    +  1  .  #    �  �  �  f  (  �  �  g  �  }  B  7  *      �  �  �  �  l  J  $  �  �  �  �  �    D                "  %  (  ,  0  5  :  B  S  c  s  �  �  �  A  G  M  L  J  B  8  '    �  �  �  �  �  `  <  �  }  &   �  �  �  �  �  �  �  �  t  ^  F  *  	  �  �  �  m  F    �  �  k  �  
2  
�  �     Z  �  �  �  n  <  �  �  
�  
/  	  �  �    �  �  �  v  B    �  �  E  �  �  �  a    �  S  �    �  T  
�  (  I  R  ;    
�  
�  
X  
  	�  	�  	K  �  �  �    �  �  �  �  �  �  �  n  H    �  �  �  H  
  �  k     �    �    t  0  ,  (  "          �  �  �  �  �  �  }  �  �  �  e  E  )  !  $  3  1  *      �  �  �  �  w  C         �  �  �  	�  
  
\  
�  
�  
}  
e  
?  
  	�  	g  	  �  8  �    G  5  �  �  	�  	�  	�  	�  	�  	�  	�  	�  	d  	8  �  �  J  �  M  �    I  A  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  l  d  �  �  �  �  x  l  a  W  L  C  8  +      �  �  �  �  �  n  O  0  v  �  �  �  l  R  9      �  �    *    �  �  *  �  �  %  6  0  *  $                    �  �  �  �  �  �  n  �  �  �  �  �  �  �  �  �  �  �  �  �  x  o  e  [  Q  H  >  �  �  �    z  u  p  f  Z  N  B  4  '      �  �  �  �  �  O  [  g  s  t  t  s  p  l  h  ^  N  >  +    �  �  �  �  �  Z  W  U  R  P  N  L  J  K  O  T  Y  R  E  7  )  �  �  �  D  �  �  �  �  �  �  w  ^  @    �  �  �  D  �  �  ,  �  �  �  r  j  a  [  V  O  D  9  +      �  �  �  �  p  C    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^    �  R  �  F  �  �  �  �  �  �  �  ]  (        �  �  "  B  C  D  O  Z  I  J  6    "  N  n  ]  G  -    �  �  �  M    �  �  x  �  �  �  �  �  �  }  w  r  m  h  c  ]  W  O  H  @  �  �  _    T    |  �  �  �  �  �  r    �  $  �  �  �  �  %  	k    �  '  ;  [  O  7    �  �  �  �  �  c  ;    �  �  j    �   �  c  O  ;  '      �  �  �  �  �  �  v  c  P  =  *      �  1  ,  (  #      �  �  �  �  �  s  H  $  �  �  �  -  �  :  �  �  �  �  �  �  �  �  u  Q    �  q  �  w  �    H  �  E  �  �  �  �  �  �  �      1  5  *        �  �  �  �  �  G  6  &      �  �  �  �  �  o  T  4    �  �  �  �  �  �  �  �    )  ?  D  @    �  �  �  S    
�  
  	W  x  D    �  T  Q  k  �  �  �  �  �  �  �  �  �  o  C    �  g  �  S  �  4  �  �    F  |  �  �  
      �  _  �  �  �  M  �    
e  �  �  �  �  �  �  �  �  �  {  a  F  *    �  �  �  �  v  V  �  �  �  �  �  O    �  �  @  �  �  F  �  �  5    �  X  �  �  �  �  r  E    �  �  j  E  /    �  �  b  "  �    :  �  �  �      .  '         �  �  �  �  �  �  �  �  �  �  w  	�  
  
7  
P  
Q  
G  
4  
  	�  	�  	�  	a  	  �  R  �  �  �    9  �  �  �  �  �  �  �  x  g  U  F  9  -      �  �  �  �  t      �  �  �  �  �  W  "  �  �  a    �  u    �  j  �  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  U  ?  1  1  2  2  3  3  4    �  �  �  �  �  e  B    �  �  �  p  a  W  M  <  -  H  �  
H  
V  
B  
;  
=  
N  
V  
U  
L  
1  	�  	�  �  n  �  @  �    ~  �  �  �  �  �  �  n  S  6      �  �  �  �  v  W  8  %    