CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�n��O�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N&�   max       P�t2      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =��w      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @F�Q�     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�X    max       @vw�
=p�     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @O            �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�@        max       @��`          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >Kƨ      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�*c   max       B/��      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��O   max       B/��      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?j�   max       C�m�      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?oZn   max       C�h.      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N&�   max       PRj      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�8�YJ��   max       ?�s����      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       =�F      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F�Q�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?׮z�H    max       @vs
=p��     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�@        max       @�@          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?�s����     �  V�      #         	      x            '   	         D               (            "                     -   	   1      -                  .                     C      !      7   =   	   �                  #      N�jaO��@N+ЇN]�6O(F�O=��P�t2N\KN��EOk�P�>N�� N��N)Z�PPL�O�ANu�"N'`fO�mO�\�N��O��O��jOn2�NeJdNmGPRjO��Nq�ZNs4<O�kfO.��O�s$N5��O���N�s.O�s�N�%�NR�lO>�GO��N9stNo �Nl�N���O�kN���O���O(��O�c�N[��PrP��N�J�O߶/N(��N�M�NB�pPT`NNP4OMD�O\hN&㱼�t���C��u�e`B��o:�o;��
;ě�<o<o<t�<#�
<#�
<D��<D��<e`B<u<�o<�t�<�t�<���<���<���<���<���<��
<��
<��
<�1<�1<�j<ě�<ě�<���<���<�/=o=t�=t�=t�=�P=��=�w=#�
=#�
='�=,1=,1=,1=0 �=0 �=<j=<j=<j=P�`=Y�=]/=�7L=�\)=���=���=���=��w��������������������^^gnrz����������znf^rot|�����trrrrrrrrrr
#(04530#
��������������������������������������������)73>JNF5����TU[`gmttttg[TTTTTTTT&%"!')-57955575)&&&&���������������������):YWNIB5)")/;BHQOMHC;/'"fcfimrz}��zmffffffff��������������������-.*B[t���������t[OE-)5BN[ampi[N=5)����������������������������������������������
#,1-.#
����������������������������������eht���������������te����
/=HNL4#
��! #/<HLUanxnfaUH;/!$)01)#)5?BDB751)cct���������������mc������ !�������������������������
#+/1/#
���oor��������������{to������
 "!!
�������������������������TPOU]anronaUTTTTTTTT55;BO[^gmqtrnh[OB;85���
#"#+,#
�������������������������������������������������������������������
#&##$!
���������5BNSH<)�����������������������

����������������������������������������		')25;BFEB951)#	�������������������������#����%#%)6BO[agc[YOHB<6)%6BOYbhk[OD6)IOS[^hhktxtsh_[OIIII����������������������������$-20)��������������������������������
!
����'*144*%	
#*020+$#
				43<>HLTOH<4444444444}����!-2/�����z}������������������������������  ����������

�����������������������������������������������E�E�E�FFFFFFE�E�E�E�E�E�E�E�E�E�E�ÓàçìöìàÓÇÑÓÓÓÓÓÓÓÓÓÓ�����������������x�s�t�x�{�����������������������	��� ���	�����������������׻����û˻ջܻԻû������������������������0�I�i�{Šŭ������Ŕ�U�#��������������0�������������������������������������������ʾ׾������׾˾ʾɾ��������������׾���	��$�+�.�1�0�"��	����׾Ծоо׿`�y�������Ŀ��������y�m�G�;�2�-�,�3�T�`�a�e�m�q�w�z�~�z�m�a�Y�T�M�O�T�Z�a�a�a�a�g�s�������������s�g�[�_�g�g�g�g�g�g�g�g���������������������������������������Ҿ������ɾ;Ǿž�����f�P�R�K�K�E�D�M�Z���[�h�t�|ĉĘĤĩĪĦĚčā�|�v�[�T�P�W�[�A�M�Z�c�d�Z�W�M�I�A�7�;�A�A�A�A�A�A�A�A����!��������������������������*�-�-�)����������������Ѽ�������̼׼ؼּʼ������r�f�Y�J�N�Y�r���������������������������{�z�����������"�/�E�C�;�A�;�/�"���	����������	��m�y�����������y�`�G�;�"����.�;�G�\�m����������	����������������������a�i�f�a�^�U�H�C�B�H�U�U�a�a�a�a�a�a�a�a�Z�^�f�k�r�f�^�Z�X�M�L�M�P�U�Z�Z�Z�Z�Z�Z�A�Z�p�~�������s�f�B�4�(�����"�'�4�A�.�;�G�T�`�q�x�q�m�`�T�G�;�.�"�	���"�.�����������������������������������������H�I�T�Z�a�b�a�\�T�O�H�A�?�H�I�H�G�G�H�H�6�B�O�k�q�h�]�O�6�)����� ����
�� �6�Z�f�j�s������{�s�f�Z�M�A�9�<�@�A�M�V�Z��'�5�A�I�I�Q�L�3������Ϲù��ùй����)�3�)�%� ����������������ʼּ����ּʼ���������������������¦¬²½¸²¦£�������ѿ���������ѿ����������������/�<�H�J�M�P�H�<�/�#�"��#�/�/�/�/�/�/�/�����������������������z�����������������A�M�Z�f�y������������s�Z�M�?�7�3�4�:�A��(�8�C�C�>�-�������������������!�"�!��� ���������������6�C�O�\�h�t�h�\�U�O�C�?�6�1�6�6�6�6�6�6�����������������������������������������/�<�H�U�V�U�T�H�C�<�/�'�#�!�#�$�/�/�/�/�����������������ƳƣƎ�~�u�jƁƏƭ���O�[�h�i�h�h�[�R�O�B�6�*�0�6�:�B�C�G�O�O�-�:�F�^�h�m�i�_�S�F�-�!����������-�_�l�x�~�����}�q�l�_�S�G�F�A�F�J�S�Y�^�_�r�~���������ɺú������������r�^�W�[�e�r�'�*�4�>�@�M�M�M�L�@�6�4�-�'�%�#�'�'�'�'ìù�������������àÇ�z�]�I�U�`�nÇì���
��/�D�?�<�9�!�
��������¾��¿������y�������������������y�u�u�o�y�y�y�y�y�yDoD�D�D�D�D�D�D�D�D�D�D�D�D�D{DnD`DYDbDo�y�������������y�l�j�l�u�y�y�y�y�y�y�y�y����!�'�,�!������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D߼4�@�I�>�C�C�@�4�'����ʻƻȻڼ��'�4�����������������������������������������5�N�Z�h�m�l�f�h�r�g�a�Z�N�A�6�5�)�)�0�5�������������������������������������w��ǡǭǭǲǭǡǡǔǑǐǔǞǡǡǡǡǡǡǡǡ : # C { m  ` % n 5 H D I v . e & < > ? > R @ A J G : = T r * " I [ + B S ; P N U 0 p Z  d t / M R | h 4 Q : = 3 4 d S L e @    �  �  H  �  �  �  �  q  �  �  �  �  �  S  _  p  q  N  �  �    �  <    �  �  �    �  �  �  o  %  _  �  �  �  �  z  �  ;  W  �  �  �  4  �  �  �  �  �  '  �  �  �  @  �  ^  ;  n  �  :  F���
<�C��t��t�;�o<���>   <e`B<D��<�t�=@�<���<�C�<u=���=�w<�9X<���=0 �=ix�=t�=,1=@�=T��<ě�<���=+=@�<���<ě�=�7L=+=��<�h=�hs=49X=q��=@�=�w=m�h=���=8Q�=<j=49X=m�h=�t�=@�=�;d=u=���=H�9=���=��=]/>Kƨ=ix�=q��=�1>H�9=���=�G�=��=�{BOQB�2B
&�B$�BP\B"JmB�B	TB��B!|JB��A�*cA� UB��B&�B�_B�4B-:B�sB"DBWJBiB��BτB�JB��B��B��BV�B@9B4;B�&B��B.�B��B�1BB�EB�B$1�B�,B�B�B=�B�7B��BC�B��BXB�Bb�B'�B�B,�yB�B/��B$��B �B�{B�|BjBNTB��B>�BK%B
=WB%>�BJ�B"@ B��B	<<B��B!x#B�:A��OA�s�B��B�B;OB�kB?�B� B!BH.BNkB �B��B��B��B
B�BEOB9YB��B��B�B��B�B��B�*B�B�\B$@B�#BПBffB@
B��B�B�5B@�BQ	BE�B��B5�B�dB,˕B:MB/��B$��B�RB JBՇB��B�jB��A�.�C�m�A�V�@�4-A��@�SbA�<A��AR�CAY�Ak�hA��A��1A���AF�rA�BRA=X�A��9A��h@���AH�hA�xUAi%$Aғ�AŎ�A?�9A<*AcU�BQ
A��5A�_�A?�)?j�A��(@���A�WAyC�A�K�A[�A?w�A�n@apBGA��fA�XvB�]A�7@w�@�F�@��@��A��7A�d�Af{C�ͪAMA
,uC�3�@�4�A��A�~�A�בB��A�s�C�h.A�d@���A�r�@���A�{}A�wtAS�`AY�sAi-A��A�p�A��6AG4A�r�A>ȃA�a(A���@�EAH�?A�u�Ai	wA҂�A�A>��A;�AcrBWpA��5A�y�A=�?oZnA��$@���A�~�Av�{A³�A��A?X5A���@cDtBcA�v�A�F�B��A�?p@t@�l�@��@σuA�5�A�}�AsC���A��A	�sC�7�@��,A�k�A��1A�rVB½      #         	      y   	         '   
         D               (            "                     -   
   2      .                  /                      D      "      7   =   	   �                  #                           E            '            -               '      !   '            )            !      %                        %               %      !      %      -   '      !            4                                 !            %                                                )            !                                                         %      +                                 N�jaO�N+ЇN]�6N�CYO=��O�M�N\KN��EO4�O���N���N��N)Z�O���Opf"Nu�"N'`fO�� O�iXN���O��OxLN��N4��NmGPRjO�Nq�ZNs4<O��O��Oko�N5��OT��N7�O�s�N�%�NR�lO=�O��(N9stNo �Nl�N���O`��Nr)8OR<|O(��O�c�N[��P�FO�Ny�Od�6N(��N�M�NB�pO��NP4O1��O\hN&�  �  �      m  `  
#  �  3  =  �  �  �  �  M  �    M  l  �  q  �  �  X  �  �  �  :  �  �  �  �      	  �  ?  �  �  �  �    o  �  �  {  J  	�  �  �  �  ;  �  �    �  +  �  �  �  n  �  ���t���o�u�e`B��o:�o=m�h;ě�<o<#�
<D��<49X<#�
<D��='�<�o<u<�o<���<���<��
<�1<�<��<��
<��
<��
<�h<�1<�1<���<���=t�<���=+=+=o=t�=t�=#�
=D��=��=�w=#�
=0 �=D��=0 �=}�=,1=0 �=0 �=D��=�C�=@�=��=Y�=]/=�7L=�F=���=��w=���=��w��������������������ljlnpzz���������zwnlrot|�����trrrrrrrrrr
#(04530#
���������������������������������������������)38:94)���TU[`gmttttg[TTTTTTTT&%"!')-57955575)&&&&��������������������)6VRLFB5) "+/;FHKMKH@;/*"    fcfimrz}��zmffffffff��������������������JGGJO[h��������th[OJ)5BN[_jmgcNK<5)����������������������������������������������
#'/,-'#
�����������������������������������ikw���������������ti���
#/6<ABA</#
�)'&./<AHPUYURHB<7/)) )/0)#)5?BDB751)cct���������������mc�������������������������������
#+/1/#
���qpt|��������������tq�����
! 
������������������������TPOU]anronaUTTTTTTTT>:88>BO[cinqokh[SOB>��

������������������������������������������������������������������������

���������).589-)����������������������

����������������������������������������)5:?BBBA=85)(������������������������������%#%)6BO[agc[YOHB<6)%6BOYbhk[OD6)IOS[^hhktxtsh_[OIIII���������������������������!$"��������������������������������
�����'*144*%	
#*020+$#
				43<>HLTOH<4444444444�����������������������������������������  ����������

�����������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�ÓàçìöìàÓÇÑÓÓÓÓÓÓÓÓÓÓ�����������������x�s�t�x�{�����������������	������	�����������������������������û˻ջܻԻû�������������������������#�0�<�G�R�U�N�A�0�#�
�������������
��������������������������������������������ʾ׾������׾˾ʾɾ������������������	���&�*�'�"��	�����ھ׾Ӿ׾㿆���������������y�`�G�;�5�0�0�7�T�`�y���a�c�m�p�v�z�{�z�m�a�\�T�P�Q�T�]�a�a�a�a�g�s�������������s�g�[�_�g�g�g�g�g�g�g�g���������������������������������������Ҿ������������������������s�c�Y�Y�`�s��h�t�yĆĖĢħĩĦĚčąā�x�t�[�U�Q�Z�h�A�M�Z�c�d�Z�W�M�I�A�7�;�A�A�A�A�A�A�A�A����!��������������������������)�+�+�)����������������Ѽr������������ɼμμʼ��������u�f�[�Y�r�������������������������������|�|����"�/�;�B�A�;�4�/�"���	� ����������m�y�����������y�p�`�T�G�>�4�0�;�G�T�X�m����������������������������������a�f�d�a�\�U�H�D�D�H�U�Z�a�a�a�a�a�a�a�a�Z�^�f�k�r�f�^�Z�X�M�L�M�P�U�Z�Z�Z�Z�Z�Z�A�Z�p�~�������s�f�B�4�(�����"�'�4�A�"�.�;�G�T�Y�`�h�c�`�T�G�.�"������"�����������������������������������������H�I�T�Z�a�b�a�\�T�O�H�A�?�H�I�H�G�G�H�H�6�B�P�[�e�m�h�Z�B�6�)��������%�6�f�g�s�������y�s�f�Z�M�A�;�=�A�B�M�Z�f�'�)�3�9�>�9�3�'�������ܹչܹ����'���)�3�)�%� ��������������������ʼּܼ��ּʼ�������������������¦¦²µ³²¦¥¦¦¦¦¦¦¦¦�������ѿ���������ѿ����������������/�<�H�J�M�P�H�<�/�#�"��#�/�/�/�/�/�/�/�����������������������z�����������������A�M�Z�f�s�s�������s�f�Z�M�D�A�;�8�A�A���(�5�8�;�7�(�������������������!�"�!��� ���������������6�C�O�\�h�t�h�\�U�O�C�?�6�1�6�6�6�6�6�6�����������������������������������������/�<�H�R�Q�H�@�<�/�)�$�'�/�/�/�/�/�/�/�/����������������������ƳƧƚƔƎƎƚƳ���O�[�f�f�[�P�O�B�6�-�6�B�F�I�O�O�O�O�O�O��!�-�:�F�O�^�`�V�S�F�-�!��������_�l�x�~�����}�q�l�_�S�G�F�A�F�J�S�Y�^�_�r�~���������ɺú������������r�^�W�[�e�r�'�*�4�>�@�M�M�M�L�@�6�4�-�'�%�#�'�'�'�'ìù���������������àÇ�z�a�U�c�nÇì�����
��#�%�(���
��������������������y���������������y�v�v�q�y�y�y�y�y�y�y�yD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDlDoDqD{�y�������������y�l�j�l�u�y�y�y�y�y�y�y�y����!�'�,�!������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D߼��'�1�4�6�4�,�'��������޻߻��������������������������������������������5�A�N�Z�g�l�k�g�d�d�Z�N�I�A�8�5�+�*�2�5�������������������������������������w��ǡǭǭǲǭǡǡǔǑǐǔǞǡǡǡǡǡǡǡǡ :  C {    % % n : E C I v   i & < > @ : K K  W G : 8 T r % " B [ * C S ; P > I 0 p Z ' = ~ . M R | b ( Q 2 = 3 4 3 S + e @    �    H  �  �  �  �  q  �  �  R  �  �  S  �  9  q  N  `  A    z    �  f  �  �  i  �  �  �  P  �  _  �  ^  �  �  z  5  N  W  �  �  �  �  �  �  �  �  �  �    �  �  @  �  ^  �  n  w  :  F  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  �    f  K  +  	  �  �  ,  �  y  ~  �  �  �  �  �  �  �  �  �  �  �  �  X    �  C  �  �   �            �  �  �  �  �  �  �  �  �  �  �  �  u  h  [      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  q  %  +  0  :  F  R  _  l  l  k  d  W  H  ,    �  �  �  I    `  V  H  5  !  
  �  �  �  �  d  ?      �  �  �  c  (  �    k    �  �  	a  	�  
  
!  
  	�  	�  	~  	  �  �  �  
  �  �  �  �  }  p  b  S  D  4  '      "  +  .  1  1  0  /  ,  )  3  )      	            �  �  �  �  �  �  �  �  }  k  3  8  ;  <  =  ;  7  0  %      �  �  �  �  �  �  h  M  2  �  �  �  �  �  �  �  �  �  �  w  K    �  �  �  m  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  H    �  �  �  �  �  �  �  �  �  �  �  �  {  o  a  Q  @  .      �  �  �  �  �  z  n  c  X  M  F  A  <  7  2  -  #    �  �  �  �  �  �  �  5  �  �    .  D  M  J  6    �  p    �    R  {  R  �  �  �  �  �  �  �  ]  1  �  �  ^  )  �  �  k  3  �  �  �      	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  M  H  C  >  9  4  /  &        �  �  �  �  �  �  �  �  �  g  k  b  U  D  1    �  �  �  �  d  G  "  �  �  �  �  s  8  P  o  ~  �  �  �  �  ~  o  [  3  �  �  �  q  H  	  �  �  �  o  q  i  W  C  )  
  �  �  �  �  y  b  =  �  �  9  �  +   �  �  �  �  �  �  �  p  Y  :    �  �  �  �  �  �  a  0    z  �  �  �  �  �  �  �  �  �  �  �  �  s  &  �  V  �  i  �  �  �  T  �  �  %  F  U  X  L  .    �  �  _    �  �  +  Q  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  v  p  k  e  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  M  6       �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  Q  A   �   �   f  �      )  4  9  :  9  6  .      �  �  y  '  �  C  �   �  �  �  �  �  �  �  �  �  �  l  U  =  )      �  �  �  y  O  �  �  �  �            �  �  �  �  �  �  �  �  �    o  g  }  }  m  \  F  3  "    �  �  �  �  �  S  �  �    r  �  �  �  �  �  �  �  �  �  �  �  s  b  P  ;  "    �  �  0   �  �  1  �  �        �  �  �  �  �  S    �  I  �    (  @    �  �  �  �  �  �  �    s  l  k  j  i  h  g  f  e  e  d  �  	  	  	  	  �  �  �  �  I  �  �  6  �  L  �  d  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  I  )  	  �  ?  ;  5  /  "       �  �  �  {  T  *  �  �  l    �  $   �  �  �  �  �  �  �  ~  r  i  ^  P  @  .      �  �  �  �  �  �  �  �  �  �  �  �  v  m  c  Z  Q  G  7     �   �   �   �   �  T  ~  �  �  �  �  }  n  [  D  *  
  �  �  x  (  �  B  �  $  O  l  �  �  �  �  �  w  W  0  �  �  ~  .  �  D  �  �  �  ,       �  �  �  �  �  �  �  �  �  �  �  x  [  3  
  �  }  1  o  _  P  @  1  !    �  �  �  �  �  s  V  :      �  �  �  �  �  �  �  �  �  o  ]  K  8  0  2  5  7  9  G  Y  k  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  E    �  j    �  a  �  �  �  4  `  v  z  p  ]  B    �  �  �  i  "  �  (  {  �   �  H  H  I  I  I  D  @  ;  6  0  )  #         �  �  �  �  �  	D  	v  	�  	�  	�  	�  	�  	�  	�  	�  	a  	  �  -  �  	  n  �  �  �  �  ~  t  j  _  R  B  /    �  �  �  �  _  #  �  �  �  ,  �  �  �  �  �  �  �  �  t  [  ?    �  �  ~  1  �  �  Y  �  �  �  �  r  e  h  m  s  g  R  =  %    �  �  �  �  v  \  C  *  �  :  6  2  8  ,    �  �  �  L    �  �  +  �    �    �  �  �     S  �  �  �  �  �  �  n  2  �  �  _  �  c  �    H  �  �  �  �  �  �  �  �  �  �  �  �  c  C  #  �  �  m  3   �  �  �  O  �  �  �  	      �  z    Z  \  ;    �  j  
�  c  �  �  �  �  �  �  |  i  V  C  /       �   �   �   �   �   �   n  +  #        �  �  �  �  �  �  �  �  ~  j  V  D  3  "    �  �  x  R  *     �  �  s  4  �  �  w  8  �  �  �  F    �  ]  �  �  4  |  �  �  �  �  �  j  !  �  R  
�  	�  �    �  �  �  �  �  �  �  �  �  m  T  <  %    �  �  �  �  �  �  �  t  �  S  ]  F  .    �  �  �  h  ,  �  �  Z     �  &  �  �    �  h  K  #  �  �  �  l  `  J  %  �  �  &  �  �  3  �  �  >  �  �  �  z  f  R  =  (    �  �  �  }  N    �  �  �  M  