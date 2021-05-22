CDF       
      obs    L   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�&�x���     0  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�?   max       P��     0  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       <�o     0      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\(��   max       @FFffffg     �  !<   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����T    max       @vo
=p��     �  -   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�          0  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �
=q   max       <T��     0  :�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B1T     0  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�I   max       B1@j     0  =$   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Ұ�   max       C�T�     0  >T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��-   max       C�H+     0  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          C     0  @�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?     0  A�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?     0  C   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�?   max       P��     0  DD   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��E����   max       ?�l"h	ԕ     0  Et   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��Q�   max       <�o     0  F�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\(��   max       @FFffffg     �  G�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vn=p��
     �  S�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            �  _�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�          0  `,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�     0  a\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�bM���   max       ?�l"h	ԕ     �  b�      	      
               8      3                                             ,         2                        ;                ,         "                     C         %                      	   
         	      4             	            .Ns0�N�a�O+�N���NR:JOZ�7O?{�M�?P
BM�z�P]��M��O�y�O&�qN5mN�b�O�?`N�=}N�[xO�XN��NxZ�O�bO��pN�w8O��HO��O�5]Ph&�O(�hP�N��O���N5�O�1�O?0<P��O�D O�
O	�N��4P3�$O=�rN�-O�UAO��N��O qqNY�$O��OÄ�P$��N���N�p%O�%�N&�YOh@>NdRVN�;OD�	NM�N�z�N�RsOa��O��YO�|NM83O��1OVO�g&N�1N��N�l%M�AN�9�O��L<�o<T��;ě�;o;o:�o:�o�D����o��`B�o�#�
�T���T���T���u��o��C����㼬1��1��9X��9X��9X��9X��j�ě��ě��ě�����������������������������������/��/��`B��h�����������o�o�C��\)�\)�\)�t��''49X�49X�<j�<j�L�ͽT���Y��]/�]/�]/�q���u�}󶽃o��7L��\)��t���t����T��-��Q�BCKOW\^g\[OCAABBBBBB2<ITUbhhbbULIF<82222?BHN[ggomjg_[NNDB?>?MUagnrtnjaZUNIMMMMMMwz}������zvswwwwwwww%)5BHNROQXXNB5)!����
#'(
������aanntnla^^aaaaaaaaaa\fihnnz}����zmaTONQ\#/3<<<0/.&#jx�������������tlfdjenz����zneeeeeeeeeee���������������������������������������������������������������������������������������4<HUY\VUH<<144444444mnqyz�������zpnkhjmm����	�������9<BHOU\UOH<<99999999�������������������������������������������������������������������������������
,/&"
����������
#/*# 
�����+:B[hy}wwhXKJB6/+''+	#In{������{UI1#		NUamqz�����zumfa]TPNam�������������zma\a��������������������JNX[gt��������tgNFEJ����������������������������������������%)9BO[^_^^hlh`[B6)%���<VdcUI0
������������������������������������������������)59BDFEB54)!��������������������e����������������tbe��������������������!#(-/7<DHGE<5/-#!!@EOZht���������t[QC@��������������������ABO[hlsnh[YOJBAAAAAAotu�������������tpoo������������������������������������}{~�\k{������������tlc[\����� �������������������������������������������������������&)$������ 
 #%#
          ?EQY[`hlvvssh[WOB=6?#02570#;<ILRUWZ\^ZULIC?<;:;z���������������ztpz6<>HLUUWUH@<79666666stz�������������{utsABNQ[\_`[NHDB@AAAAAA#+5BHIQIG<0#
	Zbjn{���������{ph[VZ�����!������mnz�����{zqnmmmmmmmm��������������������=BN[ggtqig][RNFB><==��#&'%"
�����3<BHJLLHF<<433333333����������������������������������������)+*)9<EHLPTQH<8778999999)/36<FJKC6)�ʾɾ������ƾʾ׾�����׾ʾʾʾʾʾʼ����������������ʼʼҼּؼּϼʼ����������������z�������������ĿƿſĿ����������A�<�<�A�I�M�Z�c�f�k�f�d�Z�M�A�A�A�A�A�A�����������������������������������������Z�N�A�>�6�3�5�9�A�N�Z�g�s�~���}�y�s�g�ZàÊÓÕØÛàìù��������������ùñìà�������������������������������������������������ùëù������6�g�{Ā�q�[�O�6������������������������������������������s�N�A�;�:�T�g�s�����������������������s�H�F�G�G�H�U�U�W�W�U�H�H�H�H�H�H�H�H�H�H�0�)�$����	��$�0�=�I�P�X�Z�Y�V�I�=�0�{�n�n�n�r�z�{ņŇŔŠŭŵŭŨŤŠŔŇ�{�z�p�n�g�f�n�v�zÇËÇÀ�z�z�z�z�z�z�z�z�����������������������ľ����������������/�
��������
��/�<�U�b�n�|Á�z�a�H�<�/�#����#�/�<�?�>�<�<�/�#�#�#�#�#�#�#�#�z�y�z�������������������������������z�z���׾ɾ˾׾����	���.�;�>�5�.�"���B�<�6�3�6�?�B�E�O�S�P�O�B�B�B�B�B�B�B�B�/�,�/�;�=�H�T�`�a�a�m�u�m�f�a�T�H�;�/�/��������ܻۻܻ�����'�4�9�:�4����G�@�8�6�:�S�`�m�y�~��������������y�`�G�ѿǿĿ������������ĿͿѿٿݿ����ݿ�E�EqEPEIE4E5ECEPE\EiEuE|E�E�E�E�E�E�E�E���������������������������!����������������"�.�<�G�`�m�~�w�m�G�;�.������r�l�p�u�����������������������������	���������	�� �"�'�/�8�;�?�;�7�/�"��	ƘƋ�h�\�R�C�C�O�_�hƁƚƭƮƱƸ����ƳƘ�O�L�C�9�6�.�6�C�O�U�\�h�k�m�h�\�O�O�O�O������������	��"�'�,�2�0�+���	���*� ����"�*�/�6�?�6�-�*�*�*�*�*�*�*�*����������	��"�.�;�?�G�M�P�H�;�:�.�"��f�a�M�C�A�I�M�Y�f�r��������������|�r�f�ɺ��ĺ��!�-�:�`�k�c�s�����l�e�S�-����;�/�"���
��#�/�;�H�T�[�b�a�[�T�P�H�;���������������������������������������������������������������������������������m�l�k�m�s�z���������������z�m�m�m�m�m�m��ķıģħĳ��������&�.�"�$�:�0�,������t�m�h�[�O�J�E�O�Y�[�h�tčđęĚĥĢč�tE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��н����������ý�����-�1�'�"�����к@�8�8�;�:�7�@�L�r�~���������~�r�e�Y�L�@�������������ûлԻһлƻû������������������������~����������������������������r�p�r�v���������������r�r�r�r�r�r�r�r��ӿ̿ʿѿݿ������(�2�.�(�������Ň�{�n�b�K�K�V�nŇŔŭųŲŷŷŵŭŠŔŇ�r�g��������ּ����&�'� �����ּ����rà×ÓÐÌÑÓÔàçìùùùïìàààà�������������������������
���
�������Ŀĳęčā�z�w�zāčĚĦīĮ����������Ŀ�ѿ̿ϿѿԿݿ���ݿѿѿѿѿѿѿѿѿѿѹϹù������������ùϹܹ������������Ͻ������������ĽͽʽĽ����������������������޽��������(�4�9�4�2�(�������� ��"�/�;�D�H�J�T�a�d�f�a�_�K�H�A�;�/� �׾վ׾���������������׾׾׾׾׾����������������!�)�2�5�7�5�)���6�/�2�6�@�B�O�[�b�b�[�O�H�B�6�6�6�6�6�6��ܻڻػܻ��������%�.�1�'����������������4�@�M�S�V�K�@�4���������������	���(�,�3�,�(��������޹�����	��������������x�n�_�P�R�l�����������û߻ۻл��������x�/�(�#�&�/�0�<�H�U�X�a�c�a�a�U�P�H�<�/�/����ּӼܼ������.�?�G�S�P�D�:�.��(�$�(�0�5�A�N�T�N�B�A�5�(�(�(�(�(�(�(�(�<�:�<�B�H�U�]�V�U�H�<�<�<�<�<�<�<�<�<�<�����������������������ĺǺ���������������������������
�����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��A�;�4�7�A�N�Z�g���������������x�g�Z�N�A 9 c $ - A # @ A � v 7 z 4 ? P M w 7 W Q 6 � l P ^ r B S 3 2 Y e H V  N P , ^ J G  I - ] > n : = , < U O � o ^ 2 J i � ` G T & 7 ^ Y H $ Q k H 7 f ^ \  w  �  p  �  v  �  �    �  a  �  c  :  }  e  v  �  �  �  m  A  �  �  �  
  �  D    �  s  �  �  [  s  4  �  �  	  �  R  �  �  �    w  E  �  4  v  W  �  W  �    >  s  �  �  /  W  �    �  �  f  �  s  �  E  �  r  >  �    �  V<T��;��
�#�
��`B�D����j�49X���
�u�t��m�h�T���,1���ě���1�8Q켼j��9X�,1��/�ě��C��49X��`B��+�#�
�P�`���P�t��'��H�9���]/�H�9��{�q���#�
�,1�t�����]/�P�`��o�]/�,1�8Q�'m�h�������49X�8Q콟�w�L�ͽ����L�ͽixս�o�aG��y�#��o��������7L�����xս��
�������P���T��1��{����
=qB1TB&�1B�5BxB��BY�B�ByiA�B��B1TB�B��B��B>B!�(B��B+�B[IB��B<�A��B!wiB+��B+h7B1vB��B��B'EA� B �BM�B	��B��Bv�BV*B$I BS�B��B�:B�`BEB��B`&B��B!YB�B7 B}�B?�B
�mB-H8B��B4�B�B,B<�B%D�B&�RB��B�B
�BU-B%~vB(�B��BV�BB�+B��BӍB
�B͘Bt}B�B?B1@jB&�hB�B��BT>B?�BF"B�A�}�BD�B@�BٱBȗB��B��B!��B��B?/B��B��B=A�IB!J�B+��B+@B=�B�*B�{B&�@A�Z3BvB<iB	��B��BCB>�B#��BBB��B��B�B
�]B}�B@GB yB �B�zBGBC�B>�B
!B->�B��B?gB�{B>�B=�B%L�B&��B<#B0~B
��BFyB%�/B)@�B{B�B�1Bk#B?B�mB
�QB�BA�B�aB/eAR��@�h�Aso�A=�AϡCA�u	A̲yA���A�\A��A�Z�A��B
QnA���A��AL�cA�aFA�/�A���AZi.Aئ�A�<@��Ah��Az��C�� A�A_RUA��kA��`B�BGIA[��A��jA^�*@�z/@n��A���A���A�?iA��A�MA�.�C�T�A+d�?�}�@�j@�$C@�PVA���A���A ��A�x�A�u�A�9A{�>Ұ�A$��A2��A���AU�A��A�1_@�B?@�P�A2��?*{�@�`�A���A
v�A���Aĥa@חA�q�C�ٓA�H�AR��@�@As�A=�Aτ�A��CA̔�A�~�Aל�A�x�A��[A�[�B
J�A�A�ALk�A��A���A���A\��A؂A���@�j�Ah�Az�C��.A�N�A`_�A�	A�k�B	�BESAZջA���A^�@݌9@k��A�{?A� �A���A��dA�A�P�C�H+A+<?�}O@��@�ъ@�{A�pA�"A�Aˡ�A��A߁�A|�>��-A$>9A3�*A��AW�EA��A�p�@�n@��A3��?1A{@�E!AßnA	 �A�hOA��@OfA�q�C��kA���      	      
               9      3                                             ,         3                        <   !            ,         "                     C   	      &                      	            	      5      !      
            .                           ,      1                  !                     !      #      '   /      )                  ?               -         )                     3         #                                       #      !                                                   -                  !                           !         /      )                  ?                                             )         !                                       #      !                  Ns0�N�a�O+�N���NR:JO/�O��M�?O�єM�z�PDk�M��O�y�N�_N5mN�b�O�?`N�=}N�[xO��N��NxZ�O�bO�N���O���N�QO[�Ph&�N���P�N��OHӆN5�OY_�OʎP��O�zN�+.O	�Nr��O�,NB2)N��bOT�%O��N��O qqNY�$O�%OTf�P�^N���N�p%O�s�N&�YO"��NdRVN�;OD�	NM�N�z�No��N���O��YO�|NM83O��1OVO�g&N�1N��N�l%M�AN�9�O��L  �  �  �  �  }  �    �  �  �  �  p  I  s    =  �  9  �  �  �  �  ;  >  �  	B  �  )  u  �  Z  �  �    �  �  ;    A  �  '      �  �  �  h    c  g    �  �  �  �  �  8  	  �  ;  ?  m  c  M  E  k  �  	�  %  0  �  T  �  �  v  
<�o<T��;ě�;o;o�o��o�D����9X��`B�D���#�
�T����C��T���u��o��C����㼴9X��1��9X��9X�o��j��`B�����\)�ě���`B������������������`B�����t���`B��`B���8Q�49X�\)�0 ż��o�o�C��t��8Q�'t��',1�49X�L�ͽ<j�<j�L�ͽT���Y��aG��}�]/�q���u�}󶽃o��7L��\)��t���t����T��-��Q�BCKOW\^g\[OCAABBBBBB2<ITUbhhbbULIF<82222?BHN[ggomjg_[NNDB?>?MUagnrtnjaZUNIMMMMMMwz}������zvswwwwwwww')5BNONMNNRNB5)%���
#%&#
�������aanntnla^^aaaaaaaaaafmuz������zuma\WUV[f#/3<<<0/.&#ggn|������������tpjgenz����zneeeeeeeeeee������������������� �����������������������������������������������������������������������4<HUY\VUH<<144444444mnqyz�������zpnkhjmm������������9<BHOU\UOH<<99999999������������������������������������������������������������������������������������
%*"
���������
#*(#
 ������26BO[hipomhf[OB63/02	#In{������{UI1#		RT\akmuz}�~zma_WTTRRam�������������zma\a��������������������KO[dgt�������tgb[NMK����������������������������������������&)06;BO[\^]\`[OB6) &���<VdcUI0
������������������������������������������������)59BDFEB54)!��������������������s�������������ytnils��������������������"#+/:<DDA</#  """"""MOY[hotz�����thg[SMM��������������������ABO[hlsnh[YOJBAAAAAAotu�������������tpoo������������������������������������~|�gjt�������������ytmg����� ��������������������������������������������������������$(#����� 
 #%#
          ACIOW[ehrqnpnh[OKBAA#02570#;<ILRUWZ\^ZULIC?<;:;z���������������ztpz6<>HLUUWUH@<79666666stz�������������{utsABENP[[^`[NIEBAAAAAA #02;<>=<0#Zbjn{���������{ph[VZ�����!������mnz�����{zqnmmmmmmmm��������������������=BN[ggtqig][RNFB><==��#&'%"
�����3<BHJLLHF<<433333333����������������������������������������)+*)9<EHLPTQH<8778999999)/36<FJKC6)�ʾɾ������ƾʾ׾�����׾ʾʾʾʾʾʼ����������������ʼʼҼּؼּϼʼ����������������z�������������ĿƿſĿ����������A�<�<�A�I�M�Z�c�f�k�f�d�Z�M�A�A�A�A�A�A�����������������������������������������Z�N�A�@�8�6�?�A�E�N�Z�^�g�s�|�{�v�s�g�ZàÔÙÛÞàæìùü����������ÿùìàà����������������������������������������������������)�6�B�O�[�j�n�h�[�O�B�)����������������������������������������������s�N�>�>�Y�g�s�����������������������H�F�G�G�H�U�U�W�W�U�H�H�H�H�H�H�H�H�H�H�0�)�$����	��$�0�=�I�P�X�Z�Y�V�I�=�0�{�r�u�{�~ŇŔŠťŤšŠŔŇ�{�{�{�{�{�{�z�p�n�g�f�n�v�zÇËÇÀ�z�z�z�z�z�z�z�z�����������������������ľ����������������/�
��������
��/�<�U�b�n�|Á�z�a�H�<�/�#����#�/�<�?�>�<�<�/�#�#�#�#�#�#�#�#�z�y�z�������������������������������z�z����׾˾̾׾���	��+�.�:�<�3�.�"���B�<�6�3�6�?�B�E�O�S�P�O�B�B�B�B�B�B�B�B�/�,�/�;�=�H�T�`�a�a�m�u�m�f�a�T�H�;�/�/��������ܻۻܻ�����'�4�9�:�4����T�S�G�D�A�D�G�Q�T�`�d�m�q�y�{�y�q�m�`�T�ѿ˿Ŀ����������Ŀǿѿֿݿ����ݿѿ�E�E�EtE\EME8E9ECEPE\EuE�E�E�E�E�E�E�E�E���������������������������������"�����������	��"�+�.�:�E�Q�S�G�;�.�"�����r�l�p�u�����������������������������	����������	���"�/�3�;�3�/�"���	�	ƘƋ�h�\�R�C�C�O�_�hƁƚƭƮƱƸ����ƳƘ�O�L�C�9�6�.�6�C�O�U�\�h�k�m�h�\�O�O�O�O�	������������	��"�$�+�*�%�"����	�*� ����"�*�/�6�?�6�-�*�*�*�*�*�*�*�*�"��	�������	��"�.�;�F�I�G�?�;�.�&�"�f�c�Y�R�M�E�B�L�M�Y�f�r���������z�r�f�ɺ��ĺ��!�-�:�`�k�c�s�����l�e�S�-����;�9�/�$���"�/�3�;�H�R�T�\�\�U�T�H�D�;���������������������������������������������������������������������������������z�p�m�l�m�t�z���������������z�z�z�z�z�z����Ļľ������������������
� ���������h�`�[�Z�[�h�tāĉā��t�h�h�h�h�h�h�h�hE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��нĽ����������Ľн�����������ݽк@�8�8�;�:�7�@�L�r�~���������~�r�e�Y�L�@�������������ûлԻһлƻû������������������������~����������������������������r�p�r�v���������������r�r�r�r�r�r�r�r��տͿͿӿݿ������(�0�,�(�������ŔŇ��t�n�h�c�w�{ŇŔŠŦŪūůŭŪŠŔ�������������ۼ�����#�"��
����㼽��à×ÓÐÌÑÓÔàçìùùùïìàààà�������������������������
���
�������ĿĳĦĚčā�z�x�{āčĚĦīĮ��������Ŀ�ѿ̿ϿѿԿݿ���ݿѿѿѿѿѿѿѿѿѿѹܹϹù����������ùϹܹ�������������ܽ������������ĽͽʽĽ����������������������޽��������(�4�9�4�2�(�������� ��"�/�;�D�H�J�T�a�d�f�a�_�K�H�A�;�/� �׾վ׾���������������׾׾׾׾׾����������������!�)�2�5�7�5�)���B�;�6�3�6�A�B�O�[�a�a�[�O�E�B�B�B�B�B�B���������������$�$�������������������4�@�M�S�V�K�@�4���������������	���(�,�3�,�(��������޹�����	��������������x�n�_�P�R�l�����������û߻ۻл��������x�/�(�#�&�/�0�<�H�U�X�a�c�a�a�U�P�H�<�/�/����ּӼܼ������.�?�G�S�P�D�:�.��(�$�(�0�5�A�N�T�N�B�A�5�(�(�(�(�(�(�(�(�<�:�<�B�H�U�]�V�U�H�<�<�<�<�<�<�<�<�<�<�����������������������ĺǺ���������������������������
�����������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��A�;�4�7�A�N�Z�g���������������x�g�Z�N�A 9 c $ - A ! @ A v v 2 z 4 2 P M w 7 W T 6 � l 4 d s H G 3 ; Y e A V  ; P  \ J ( ( < $ : > n : = , 0 K O � n ^ ' J i � ` G K ! 7 ^ Y H $ Q k H 7 f ^ \  w  �  p  �  v  m  ^    �  a  I  c  :  �  e  v  �  �  �  J  A  �  �  .  �  �    �  �    �  �  �  s  �  h  �  S  A  R  t  u  ^  �  �  E  �  4  v  A  �  �  �      s  \  �  /  W  �    ~  �  f  �  s  �  E  �  r  >  �    �  V  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  w  k  _  S  E  5  $       �   �  �  �  �  �  �  �  �  �  �  �  �  }  s  g  [  Z  ]  d  r  �  �  �  �  �  �  �  �  �  �  e  A    �  �  �  P    �   �   l  �  �  �  �  �  �  �  �  s  `  K  6      �  �  �  i  v  �  }  z  w  u  q  i  a  Y  O  @  1  #    �  �  �  �  z  N  #  v  �  �  �  }  c  H  *    �  �  �  X    �  Z  �  P  �  �                    	  �  �  �  �  _  ;    �  �  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  V  �  �  �  �  �  �  �  �  _    �  f    �  %     �  [  �  z  t  n  g  a  [  T  N  H  ?  5  +  !         �   �   �  r  �  �  �  m  H    �  �  �  k  L    �  �  �  �  h  {  �  p  o  n  m  l  k  j  m  q  u  z  ~  �  �  �  �  �  �  �  �  I  A  5  (         �  �  �  �  �  �  j  6  �  �  S    �    1  >  a  r  r  p  k  c  V  >  %    �  �  v  ;  �  �  �          �  �  �  �  �  �  �  �  �  �  x  p  _  A  '    =  8  3  /  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  T  ,     �  �  �  L    �  �  �  x  �        9  1  )  "        �  �  �  �  �  x  ^  G  1         �  �  �  �  w  l  `  U  P  O  N  L  K  J  F  >  5  -  $      �  �  �  �  �  o  Y  A  $    �  �  �  Z    �  6  �  b   �  �  �  �  �  �  �  �  �  x  d  M  6      �  �  �  �  �  k  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  l  Z  H  6  $    ;  4  *        �  �  �  �  �  �  �  X  0  	  �  �  �  �      �  �      #  5  <  <  3       �  �  r  @    �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  b  N  :     �   �   a  	/  	=  	B  	=  	1  	  �  �  �  Q    �  -  �  �  �    �  �  �  �  �  �  �  �  �  y  j  T  8    �  �  �  k  8  �  �  k              !      (    
  �  �  �  0  �  Y  �  �  �  u  f  U  ;    �  �  �  o  6  �  �  p  %  �  z  7  �  |   P  �  �  �  �  �  �  �  �  �  �  v  \  C  +        
     �  Z  N  ?  .      �  �  �  �  a  =      �  �  �  [     �  �  �    ~  {  r  h  _  Q  ;  %    �  �  �  �  �  �  �  �  U  v  �  �  �  �  �  {  f  N  /    �  �  �  T    �  "  �    "  +  4  ;  6  1  ,  %        �  �  �  �  �  �  �  k  9  i  �  �  �  �  �  �  m  U  7    �  �  K  �  �  O    �  �  �  �  �  �  �  o  J     �  �  �  [  #  �  �  "  �  �  �  ;  1    �  �  z  B  )          �  �  n    �  0  �   �  _  �  �  �  �         �  �  �  f  0  �  �  &  �  9  �  �  $  8  @  >  3  &      �  �  �  �  d  (  �  �  7  �  �  )  �  �  �  �  �  �  p  U  <  $    �  �  �  W  >    �  �  �  �      &  "        �  �  �  �  �  o  X  C  .  6  I  [  {  �  �  �  �  	        �  �  �  �  �  �  m  )  �  S  +  �  
      	          	        #  +  ;  S  h  |  ^  e  o  y  �  �  �  w  a  E  #  �  �  e  %  �  �  x  Y  D  ,  v  z  u  }  �  �  �  �  �  �  �  �  �  d  1  �  �    �  0  �  �  �  �  �  �  t  M  #  �  �  �  Z  0    �  z  �  =   �  h  Z  K  =  0  ,  (  )  $      �  �  �  �  d  0  �  �  �        �  �  �  �  �  �  f  F  "  �  �  �  m  9  #    �  c  ^  X  S  N  H  @  6  ,        �  �  �  �    -  Q  t  f  f  a  U  F  5  $    �  �  �  �  R     �  �  N  �  �  �  j  �  �  �  �  �      �  �  �  �  �  U    �  Y  �  J  D  ;  |  �  {  j  O  ,    �  �  ,  �  d  �  k  �         �  �  z  s  h  [  M  <  +       �  �  �  �  �  w  a  O  A  3  �  �  �  �  �  �  ~  j  U  @  .      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  -    �  �  �  c     r  �  "   �  �  �  �  y  l  `  S  A  ,      �  �  �  �  �  j  M  0    �    /  7  6  +      �  �  �  z  q  �  �  �  G  �  z    	      �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  `  N  �  �  �  �  �  �  u  e  Q  ;       �  �  �  z  I  
  �  �  ;  ,        �  �  �  �  �  p  E    �  �  �  �  �  �    ?  <  9  6  3  0  -  *  '  $  !              �  �  �  m  g  a  Z  S  K  C  ;  0  %      �  �  �  �  �  _  >    Y  ^  b  V  H  3      �  �  �  �    c  E  (  �  �  �  �    1  '     !  1  >  F  M  J  5      �  �  �  Q  �     l  E  @  :  1  &      �  �  �  �  �  �  �  �  �  Q  �  �  /  k  Z  J  6  !    �  �  �  �  �  v  [  ?  #    �  �  �  [  �  �  �  �  �  �  �  �  �  r  d  U  J  B  9  0  *  $      	�  	�  	�  	b  	0  �  �  t    �  d    �  E  �  o  �  �    �  %        �  �  �  �  �  �  [  ,  �  �  z  9  �  �  r  4  0  !      �  �  �  �  Y  %  �  �  �  �  F    �     �  �  �  �  �  �  �  �  �  v  j  ]  Y  ^  c  h  m  Z  @  &    �  T  @  ,      �  �  �  �  b  @    �  �  �  u  C    �  �  �  �  �  �  �  �  �  �  z  \  =    �  �  �  q  B    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  ]  B  %    �  �  �  z  a  G  ,    �  �  �    0  Q  |  
  	�  	�  	�  	u  	D  	  �  _  �  �  '  �  l  [    �  [  %  a