CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�/��v�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <#�
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @FFffffg     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vs\(�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�4�           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       ��o       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�N�   max       B0KM       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�qj   max       B0;�       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >!��   max       C�|�       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >-�4   max       C�w�       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          s       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�p�       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�1���o   max       ?�_��Ft       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       ;ě�       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @F,�����     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vs\(�     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ϊ        max       @�i@           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��)^�	   max       ?�]c�e��     �  \8   E         r      2                  7   3      @                  Y   &   $      	   (      .         1         "               B               9   ;         >       '          	                  	            -   &               PlN��Nt�6P��PA��PP��O-^JN5D�NBeN�OO�1�P�p�PBNs�jP
�O"	�N�RN�BO�iN��
P��Or;LO�'jON��O���N��|P	lNMepN)�oO*�O�=9O �P�.O%BO��OK4�NT��P �,N��N��OD$O�ݴPvY�P͚N���O
�OE2�NȈ�OƳ�N��+Pe3N�MMOAyNH��Ow��N��O��N���O�Z O��Nd��O�x�O��NOۺO:�N���Np$�M��<#�
;ě�;D���D����o��`B��`B�t��#�
�D����o��o��C����
��1��1��j�ě���`B��`B��h�������+�C��C��t��t��t��t��t��t���w�#�
�#�
�#�
�''',1�,1�0 Ž0 Ž49X�49X�]/�e`B�e`B�e`B�e`B�e`B�q���q����%��%��%��%��C���C���C���\)���P�������-���-���w��9X����������!��������xz{�������|zxxxxxxxx��������������������
<]bms{yo<# ������6BUh����t[B5+��!(4EHTap������mT;*!! )6CJNOQZ\g\OC6��������������������enz��������zrneeeeee#$/<HLTNH=</#,:?CGUnz����vxnaH</,� 	#0{�����wI<0���abmsz�����������zmaalt�������}tlllllllll!0Han~����zqa_H/!!DHUYYaz���zndaUIKJFD��������������������$)*0)%��������������������:BOR[fgd[[OMGB?:::::$5BN[foppjg[[N7"����!*-55������,6B[h�����tqrh[OA6+,��������������������=BGOV[``^\[WOOMDCA>=ztnaU:46>HNUanv}��z�
#/22/*#
������hz������������znheeh���������������������������������������������
�����������+01)�������#)56;BDD>96

O[agt�����������[IBO������������������������������������������������������������)/)  N[gt����������g[HFGN��������������������������������������������� �����������1BN[t������tNB5) B[����������}tp[PF=B���
)64,#
������������

��������jnp{����������{ztnmj�
#*/30(
�����
#$%''#

���������������������������

�������
$06<IKdifUIA0	
hjgd[NB65258BN[hhhhh������� ���������z���������yzzzzzzzzz��������������������OUaajnrxz}znaWUNLMOO��������������������������������������������������������������������������������ggmkpt|����ytgggggggMSW^dgt��������t[TMMMUaez�������naUNLJKMLN[gt������tngd^[WNL�� �������Y[gstz~xtg[[RQYYYYYY������������!#/0/.)%#"  !!!!!!!!�s�X�I�E�H�N�Z�s�����������������������sÇ�~�z�r�n�zÇÓàÚÓÎÇÇÇÇÇÇÇÇ�Z�W�V�X�Z�f�j�s�w�z�s�f�Z�Z�Z�Z�Z�Z�Z�Z���y�o�l�������нݾ�(�A�a�i�e�M�A��Ľ��T�O�@�4�/�5�T�c�y�������������������`�T�	����������������������"�4�B�M�I�;�"�	��	�������	���"�'�.�2�3�5�5�/�.�"��m�c�`�]�X�`�m�n�y�{�y�r�m�m�m�m�m�m�m�m���������������������������������������������������������������������������������z�m�a�H�;�(���� �;�H�a�m�����������z���������s�X�N�2�=�g�s����������� �������5�����߿пſǿͿݿ����5�E�T�\�j�g�5�����������
���"���
�����������������A�4�%�$�(�4�A�f�s�������������n�f�Z�M�A�#��'�/�<�D�E�H�U�[�f�h�n�u�k�U�H�<�/�#�����������$�'�$������������������ìæáàßàìôùûùîìììììììì�������z���������������ûʻʻŻû��������a�\�`�a�f�n�zÇÓÓÕÓÇ�~�z�n�a�a�a�a������������5�Z�d�s�x�u�N�5���������z�x�z�~�����������������������������x�q�r�x�z���������������ûλ̻Ż������x�������������������žʾʾ׾ؾξʾþ�����������������������������������������������"�(�-�-�"��	�����Ӿƾľʾ׾��	�ìæàÛÚÝàìðùýüùùììììììƚƄ�}�s�o�j�uƎƚƷ��������������ƳƧƚ�C�=�A�C�O�\�^�h�i�h�\�O�C�C�C�C�C�C�C�C��������	���	���������������
�
����'�4�@�M�V�X�M�I�@�;�4�'���r�[�[�f�k���������������������������;�:�/�+�"������"�/�H�T�W�[�Y�T�H�;�$�	��������������0�=�I�V�c�h�W�V�I�$�����������������������������������ŔŐŔŠţšŤŠŭŹ��������������ŹŠŔ�^�T�P�Q�T�`�m�y�������������������y�m�^¿¾½¿��������������������¿¿¿¿¿¿����������������������)�,�+������������������������������������������������������������!������������������������$� �"�$�'�0�=�I�V�\�b�m�b�b�V�I�=�0�$�$��׾ľ¾žӾҾ׾���	��������	����������������������5�L�[�o�g�V�B����пп��������Ŀݿ����"�4�2�5�-��������Z�Y�N�M�A�8�A�N�Z�d�g�s�|�u�s�g�Z�Z�Z�Z�'������-�4�@�M�Y�f�k�f�\�Y�M�D�@�'D�D�D�D�D�D�D�D�D�D�EEEEEEED�D�D�E*EEE(E*E7ECEPEYE\E_E\EPEPECE7E*E*E*E*�3�(�0�&�(�:�L�r�~�����������~�r�Y�L�@�3�l�a�b�_�^�_�g�l�t�w�x��|�������~�x�l�l���������������ûл��������л������Ľ˽нսؽ׽нϽĽ��������������������������ʼԼ߼�����
��	������ּʼ��O�H�C�?�C�O�\�\�h�r�h�\�O�O�O�O�O�O�O�OĳĦĚčā�t�k�j�tāčĚĦĳĿ������Ŀĳ�5�+�5�5�A�G�N�Z�b�g�o�p�l�g�e�Z�N�A�5�5�s�r�s�v�w�|���������������������������s�0�(�#���
��
�
��#�+�0�1�4�3�0�0�0�0�~�`�V�Y�r���������ɺֺ׺ɺƺ����������~�!��������ۺ������!�+�4�3�8�-�!�U�T�I�<�0�,�0�9�<�I�P�U�Y�W�U�U�U�U�U�U����¿²¦ ¦²¿�������	�������乄�����������ùϹܹ��������Ϲù�����ĳĨĪĬĮĲĿ��������������������Ŀĵĳ�D�B�S�`�l�y���������������������y�`�S�D�n�l�g�n�p�zÇÓÕÚÓÓÇ�z�n�n�n�n�n�n�������������������������������E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�  - G 2 c ) I _ ~ u U U h L = l \ P E D E T M 5 V : _ 2 H Q - , H . ( Q H c I M F K P f  d < 5 D 0 � 6 D c e N Q T V L J R 9 G N |  P �    �  �  |  m  �  m  �  w  �      �  �  �  �  �  V  C  A  �  �  )  �  @  '  m  �    h  N  e  6  |  �  ]  N  �  �  �  �  :  =  m  �  r  �  ;  �  �  �  \  A    �  \  �  (  �  �  [  ~  �  �  �  q  �  �  �  E�m�h��o��o��C��e`B����T���e`B����\)��\)��C���/��1�t������ixս0 Ž��O߽�7L�t��'���8Q콧#�
�'�1��+�49X��t��q���T���Y��@�����D���Y��aG�����������L�ͽ}�����9X�����t���-��C���t���O߽�j�������T����Ƨ�����������`B��9X��j��E���j���B&�B neB�B%��B
`A�N�B0KMBBY�BvCB�B&r�B ~hB�`B��B�7B��B�BvB��B�
B()B*OB �,B��B%�BM�B �B��B�B$B�FBFWB	�]B��B%B+~B��B	͓B�Bm�B��B��B
�>BWqBd�B(�B%.B/lB!�iB"�YB%BD2B-4EB
�B�-BP�B��B��B�GB|KB	ɓB	��B�B	�B%+B	=�B�MB��B=�B ��BO�B&C�B5A�qjB0;�B:cB��B\KBD�B&�B ?`B��B�DB�B��B�vB=�B��B��B˕BB�B ��B��BlB`oB ��B�B��B�6B�KB�WB	��B�bB��B+A�B�pB	cB��B��B��B	A_B	�B�B?yB)75B?�B@B!�!B"�0B%AB2�B-=oB
��B��B=�B�}BϝB��B�*B	��B	�BG�B	P>B?�B	?bBE�B�&A�T�A�wAA�A-�*Al��A��A]b�Aj:wAκA��A��A�0�A��A�*�A>Y)A�9"B��A�d�@��A�A���A��B@��JAM{�A�a�AX�eA�d�BĔBh�AY��@�?�@��A�z�B	�2A�f�A�Y�An�pA�D
A���A�ZBҷB
�AX�A���A��A���@�D�C�2=C��J?��@�|�@�}A&�[A��B�DA�ţA���A�ZVA�$�@��@_2�A�9�A�4 >!��A�ڌA�zA�pA/�	C�|�A�E�A�m�AA%�A3E�Ap��A��A\�Aj�A�y�A�[�A���A���A��jA�~�A=iAŀB	�A̤�@���Aǀ�A���A��3@���AO3A���AZ5�À	B4gBHCAY@��@�A��B	��A�y�A��Al��A���A�
�A��:B��B9AX�A�sA	OA�}}@�̯C�(C��:?�k�@��2@�S.A&��A�LB=�A��A���A���A��@G_@cٍA�+sA�n�>-�4A�A:�A��A.��C�w�   E         s      2                  7   4      @      	            Y   '   %      	   (      /         1      	   "               B               9   <      	   ?   !   '          
                  	            -   &                  )         C   3   /               %   =   )      '                  '                     #                  '      !         +            #   =   %               !      +                        !         #   !                           5   3                  %   =                           !                                       !      !                     !   1   #                     #                                                   O�TkN��Nt�6P��PA��O��JOa�N5D�NBeN�OO�1�P�p�O�q'Ns�jO��{NK�N�RN�BN��}N�y�OǓ�OL��O�=�ON��O���N��|O�m3NMepN)�oO �OZ&;O �O߄�O%BO��OK4�NT��O��N��N��OD$O��P0��O�'�N���O
�O��N��aOI�N���O� FN�MMOAyNH��OY��N��O�*N��!Oxi�O��Nd��O��$O��TOۺN��wN���Np$�M��  �  �  z  �  �     �  Y  G  �  1  q  ,  �  f    �  �  V  �  x  �  -  �  �  �  ~  q  C  �  
�  '  `  �  M  �      	9  �  �  �    �     !  r  o  	�  �  A  �  J  V  H  �  j  P    \  >  �  2  �  �  4  �  q  ��u;ě�;D����`B��o���t��t��#�
�D����o��o�C����
�C�������j�ě��C���h�@��C��C����+�C��C��,1�t��t��0 Ž,1�t��8Q�#�
�#�
�#�
�'���',1�,1�@��Y��Y��49X�]/��7L�y�#��\)�m�h�u�q���q����%�����%��o��O߽�\)��C���\)���
���㽝�-���罟�w��9X�����������	�������xz{�������|zxxxxxxxx������������������ 
#<Uelke[I0
��� ��6BUh����t[B5+��>HTamz����zmaTHC;97>"*,6CGKMOSOMC6*&��������������������enz��������zrneeeeee#$/<HLTNH=</#,:?CGUnz����vxnaH</,� 	#0{�����wI<0���krw}����������zmkijklt�������}tlllllllll",9HUans{~}zmaWH</$"[anz}�znaU[[[[[[[[[[��������������������$)*0)%��������������������;BOQ[efa[RONHBA;;;;;#)5BN[`fijjg[NE5,#�����'(!������/6B[h~���toph[OB;7//��������������������=BGOV[``^\[WOOMDCA>=ztnaU:46>HNUanv}��z�
#/22/*#
������lz������������zsliil����������������������������������������������	
�����������#*,)%�����#)56;BDD>96

LOXdgt�����������gNL������������������������������������������������������������)/)  X[gt~������tg`[UQORX��������������������������������������������� �����������$.5BN[t�����t[NB5)"$M������������tg[NHHM�����
!*.)
�����������

��������jnp{����������{ztnmj���
#+.+#!
���
##%%#
��������������������������������������#/>Za\UI60#hjgd[NB65258BN[hhhhh������� ���������z���������yzzzzzzzzz��������������������OUaajnrxz}znaWUNLMOO��������������������������������������������������������������������������������ggmkpt|����ytgggggggW[cht��������tg[QQTWLUadz������znaUTNLJLLN[gt������tngd^[WNL���������������Y[gstz~xtg[[RQYYYYYY������������!#/0/.)%#"  !!!!!!!!���s�c�Y�[�d�g�s������������������������Ç�~�z�r�n�zÇÓàÚÓÎÇÇÇÇÇÇÇÇ�Z�W�V�X�Z�f�j�s�w�z�s�f�Z�Z�Z�Z�Z�Z�Z�Z�������������ݾ��4�A�N�W�P�F�4��н����T�O�@�4�/�5�T�c�y�������������������`�T���������������	��"�-�0�6�4�1�/�'�"����"��	�����������	���"�,�.�1�2�/�.�&�"�m�c�`�]�X�`�m�n�y�{�y�r�m�m�m�m�m�m�m�m���������������������������������������������������������������������������������z�m�a�H�;�(���� �;�H�a�m�����������z���������s�X�N�2�=�g�s����������� ������������߿ٿٿݿ����6�A�G�O�H�A�5�(������������
���"���
�����������������M�A�4�+�*�*�0�4�A�Z�f�o�����x�o�f�Z�M�H�H�G�H�H�U�^�_�^�U�H�H�H�H�H�H�H�H�H�H�����������$�'�$������������������ìæáàßàìôùûùîìììììììì�����������������������ûƻǻû»��������a�]�`�a�g�n�zÇÏÓÔÓÇ�|�z�n�a�a�a�a�������������5�G�X�Z�N�@�5���������{�z�y�{����������������������������x�t�s�z�{�������������û̻ʻû��������x�������������������žʾʾ׾ؾξʾþ�����������������������������������������������"�(�-�-�"��	�����Ӿƾľʾ׾��	�ìæàÛÚÝàìðùýüùùììììììƚƉƁ�}�y�yƂƎƚƳ��������������ƳƧƚ�C�=�A�C�O�\�^�h�i�h�\�O�C�C�C�C�C�C�C�C��������	���	���������������������'�4�@�F�L�E�@�7�4�'����r�h�d�f�q�{�������������������������;�:�/�+�"������"�/�H�T�W�[�Y�T�H�;�=�0���������������$�0�=�I�W�b�^�W�I�=�����������������������������������ŔŐŔŠţšŤŠŭŹ��������������ŹŠŔ�^�T�P�Q�T�`�m�y�������������������y�m�^¿¾½¿��������������������¿¿¿¿¿¿�������������������	��
�����������������������������������������������������������������!������������������������$� �"�$�'�0�=�I�V�\�b�m�b�b�V�I�=�0�$�$���׾ȾžȾؾؾ���	��������	����������������������5�A�I�N�J�@�!�����޿ݿѿſ��������Ŀѿݿ����'�+�#�������Z�Y�N�M�A�8�A�N�Z�d�g�s�|�u�s�g�Z�Z�Z�Z�'������-�4�@�M�Y�f�k�f�\�Y�M�D�@�'D�D�D�D�D�D�D�D�D�D�D�D�D�EEEED�D�D�E*E$E%E*E5E7ECEPEUE\EPELECE7E*E*E*E*E*E*�L�@�<�<�:�@�L�Y�e�r�~���������~�r�e�Y�L�l�d�c�_�_�_�i�l�x�~����|�x�l�l�l�l�l�l�����������ûл������������л��������Ľ˽нսؽ׽нϽĽ��������������������������ʼԼ߼�����
��	������ּʼ��O�H�C�?�C�O�\�\�h�r�h�\�O�O�O�O�O�O�O�OĳĮĦĚčā�t�l�l�tāčĚĦĳ������Ŀĳ�5�+�5�5�A�G�N�Z�b�g�o�p�l�g�e�Z�N�A�5�5���z�w�x�{�}�������������������������������
��
���#�*�0�0�2�0�#�������~�w�r�b�X�e�h�r���������Ǻº����������~�!��������ۺ������!�+�4�3�8�-�!�U�T�I�<�0�,�0�9�<�I�P�U�Y�W�U�U�U�U�U�U��¿²¦§²¿��������
����������˹������������ùϹ߹��������Ϲù���ĳĨĪĬĮĲĿ��������������������Ŀĵĳ�S�M�K�S�`�g�l�y�y���y�v�l�`�S�S�S�S�S�S�n�l�g�n�p�zÇÓÕÚÓÓÇ�z�n�n�n�n�n�n�������������������������������E�E�E�E�E�E�E�E�FE�E�E�E�E�E�E�E�E�E�E�   - G 1 c ! * _ ~ u U U b L . g \ P 6 G 3 D D 5 V : _ * H Q   ( H * ( Q H c ? M F K N c ! d < 8 I  ` 6 D c e P Q O ? 9 J R 9 A N M  P �    :  �  |  Y  �  A  %  w  �      �  �  �  �  �  V  C  �  �  �  �  R  @  '  m  �  �  h  N    �  |  �  ]  N  �  �  X  �  :  =  �  �  �  �  ;  5  �  �  �  �    �  \  �  (  Z  �  �  ~  �  d  �  q  �  �  �  E  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�    �  �  N  �  �  �  �  �  �  �  �  P  	  �  5  �  �  �  �  �  �  �  �  �  z  f  Q  =  %    �  �  �  �  Q    �    *  z  o  e  Z  O  E  :  /  $           �  �  �        %  �     ]  x  �  w  U  #  �  �    �  R    �  �  �    )  �  �  �  �  ~  n  Z  C  )    �  �  �  �  �  �  h  6    �  �  x  �  �  �  �  �  �  �  �     �  �  �  2  �  &     �  W  �  �  �  �  �  �  �  �  �  t  c  Q  =  $  
  �  �  �  }  W  3  Y  O  E  <  2  '    	  �  �  �  �  �  �  f  H  )  	   �   �  G  L  Q  V  [  ^  W  Q  J  D  9  *      �  �  �  �  �  �  �  �  �  ~  q  e  X  J  <  /  "        �  �  �    8  h  1  1  &      	  �  �  �  �  �  �  �  w  J  %  �  �  �  �  q  =  �  �  G  �  �  X  "  �  �  �  t  ?    �  |  #  �   �  M  �  �  �    $  ,  )    �  �  ^  �  �  �  y  �  �  ^   �  �  �  �  �  �  �  �  u  h  Z  L  <  %    �  �  �  T     �  �    #  G  e  Q  6    �  �  �  �    P    �  p  �  q    3  M  �  �  �  �    !    �  �  �  S    �  �  l  2  �  �  �  �  �  �  �  �  �  �  �  �  u  e  T  @  -      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z        2  U  D  *  
  �  �  z  $  �  |    �  $  z  �    �  �  �  �  �  �  �  �  �  �  �  �  U    �  �  A  �  �  R    A  Q  j  w  p  T  !  
�  
�  
{  
)  	�  	.  u  �  �  �  �  �  C  �  �  �  �  i  O  +  �  �  �  D  �  �  7  �  s    �  ,  �  )  -  &    �  �  �  =  �  �  P  !    �  p    �  1  �  �  �  �  �  �  �  �  �  �  x  o  g  b  `  ^  \  G  ,     �  �  �  �  �  w  g  Q  ;  "  	  �  �  �  �  o  e  d  \  J  7  �  �  �  v  u  s  j  [  J  6      �  �  e  (  �  �  N  �  ~  |  y  s  k  b  U  F  -    �  �  �  \    �  �  <  �  8  V  [  l  n  g  X  =    �  �  �  �  Y    �  [  �  @  �  #  C  >  8  3  .  +  -  0  2  5  7  :  <  ?  A  B  C  C  C  D  �  �  �  �  �  �  �  �  �  w  i  [  `  s  �  �  �  �  x  l  
  
]  
�  
�  
�  
{  
\  
.  	�  	�  	X  �  �  
  �  �  0  U  k  n         &  &        �  �  �  �  �  a  9  	  �  x  �  �  `  X  Q  J  D  <  .       �  �  �  �  �  �  {  m  ^  K  9  �  �  �  �  �  �  �  �  k  T  :    �  �  �  I  �  D  �  '  M  J  C  7  ,    
  �  �  �  �  x  T  .    �  �  U  �  �  �  �  �  �  �  �  �  �  �  �  v  i  \  T  U  [  b  e  c  ^        �  �  �  �  �  �  �  {  p  ^  8    �  �  c  (   �    �  �  �  �  �  �  �  ~  b  G  ,    �  �  �  �    ~   �  $  L  o  �  �  	  	-  	9  	.  	  �  �  e    �    m  �  �  C  �  �  �  �  �  �  �  �  s  ]  G  0       �  �  �  �  |  d  �    ,  ]  �  	     %  *  -  0  2  3  3  2  0  -  *  &  !  �  �  �  �  �  �  f  H  $  �  �  �  y  H    �  x  F  "  	  �  �        �  �  �  S  3  $  �  �  �  W    �  A  &  �  �  �  �  �  �  �  �  [  �  �  "  �  �  �  �  C  �  �  �  �  �           �  �  �  �  N    �  h    �  /  �  5  �  �  !        �  �  �  �  t  Y  L  N  P  H  ,    �  �  �  �  r  q  p  j  a  W  H  9  '      �  �  �  �  }  ^  =     �  x  �  I  j  n  d  F    �  �  R  
�  
|  	�  	6  s  p  �  W  �  	�  	�  	�  	�  	�  	�  	�  	n  	B  �  w  �  ~  �  E  �  �      �  �  6  d  s  �  �  �  �  �  �  ^    �  �  ;  �  �    �  u  �  �  %  9  /  (  !    �  �  �  �  �  �  �  �  �  u  �    �  �  �  �  �  �  �  �  �  �  �  q  ^  @  )    �  �  �  E  J  2    
  �  �  �  �  �  �  �  �  g  B    �  �  K   �   �  V  D  =  =  "  	  �  �  �  �  �  �  u  S  4    �  �  �  �  H  @  8  0  +  &  !  "  '  ,  5  A  N  W  Z  ]  `  e  j  o  �  �  �  �  `  4    �  �  �  Y  D  %    �  (  �  �  	    j  V  B  .      �  �  �  �  �  v  \  =    �  �  ^    �  0  N  O  ?  +    �  �  �  �  w  H    �  �  G  �  �    �  �          �  �  �  �  �  �  ^  :    �  �  �  G  �  y  1  W  S  C  &    �  �  �  c  0  �  �  �  H  �  �  <  �   �  >  .      �  �  �  �  �  �  j  T  =  !    �  �  �  �  j  �  w  h  Y  J  <  -      �  �  �  �  �  �  �  �  {  i  X  (  .  1  1  -  '          �  �  P  �  �    �    }  �  �  �  �    g  E    �  �  �  O    �  }    �    z  �    *  �  {  `  @    �  �  �  �  e  =    �  �  v  )  �  �  �  K  �  �  �  �  �  �  �    +    �  �  �  �  f  &  �  o    �  �  �  �  �  �  �  �  }  ]  <    �  �  �  Z    �  �  \    q  h  ^  U  L  C  :  1  '          �  �  �        )  �  �  �  �  v  g  X  I  :  )      �  �  �  �  `  5  	  �