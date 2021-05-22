CDF       
      obs    C   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�p��
=q       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�`�   max       P��4       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?:�G�{   max       @FQ��     
x   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v_�z�H     
x  +H   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P            �  5�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�V�           6H   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��+   max       �ě�       7T   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B.<�       8`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B.J�       9l   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >S��   max       C��[       :x   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >}�}   max       C��k       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          t       <�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9       =�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       >�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�`�   max       PR81       ?�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���D��   max       ?��"��`B       @�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <D��       A�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?^�Q�   max       @Fz�G�     
x  B�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v_�z�H     
x  MP   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            �  W�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�ՠ           XP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F
   max         F
       Y\   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?��8�YJ�     �  Zh   1   &   H   5      N         !         ,   
   ^      0   1      	            
         +   )   8         6   s      !               
   
         T                           
      N      
                     8            O���PG6RPQ��O��O-��P6�fNq!O��Pl�\N�%EM�`�O�D+O'0�P��4O頝O©�P'�N��ND'!N�=�O��lN0{*O"�N퀦N�VP�xO\�PN߭�OixPq�+Pj0�O-�NOl��N3N1N_ğO2&�N(PJN�0RN�O�N���N�+�Pm��Oz OD�O0m|N��O�g�O�ONZ	�O��N���N�x�P5�]Ni+�NBwMO�f�O��N���N�"WO�?N���Oj�O���N�KQM�+M�	<���;�o;o;o;o%   ��o�o�o���
���
��`B�o�o�o�t��#�
�#�
�D����o��t����
��9X��9X��9X��j��/��h��h��h���o�o�o�+�C��C��C��C��\)�t���P��P��P��������w�',1�,1�,1�H�9�H�9�H�9�Y��aG��ixսu�u�y�#������㽟�w���w����������������� 
�����%Bhx���������t[B0&"%Jz������������mfaTOJ#)6BGKNOQPNB6-;BOR[omhf_[YVOMPA<:;���
#/<UepXB<&
�����<<=IU`ZUI<<<<<<<<<<<FHTakmnnmjaTHHGEA@AFl����������������zmlDHTagmtmjaYTQHFDDDDD
#'&#
�����#&.0,&
������?BNQU[gs����wtg[XIA?}���������������})/<Han���zvUC)15/#)mtz����������zmjhhjmlq}��������������sll���������������������������.6BORZ[][OB<61......����������	�������|�����������||||||||3;@EHTahmrtsmlaXTH;3��������������������������������������������	,?C7��������)25:=851)��)-.3BNgt~���|lB5) )lnz�����zunidbbllll�����)/1(��������������������������)5N[`eRB5������)5>ABDB?5)	AGO[hnty|~|{th[OB:AA����� �������������������������������������
#/30/*"
�����RUadlhaVULRRRRRRRRRR������������������������������������������������������������pt}���������ytnipppp�����&5AH?/���)-2:ABEDA6)^anz�������znka`\[^^����������������������������������������
)6GOTVPOB6)  agkt�������ztg^a`_aa�������������������
)241!"
������!#0<BEED@<0+%#""!!!!����������������������5IhnjkgB5����������������������������������������������������������������fg}����������tkgccff�����������������������#��������������������������TU[anwonaULOTTTTTTTT��
"$! " 
������MO[_gkt���������t[NMqt��������tpjjqqqqqqMUanongaUOMMMMMMMMMMGIPSUVZ^ZULIGGGGGGGG�/�'�
�������������
��#�<�H�U�]�_�Z�<�/�.���"�'�?�2�;�G�`�m�����������m�T�;�.��ĮĩĩĭĸĿ���������� �"�
���������̻F�;�-�$��!�-�:�F�S�l�x���������w�l�_�F�����������ûлܻ����������ܻû������(������"�7�A�M�f�����z�}�z�s�A�(�ʼ����������ʼϼѼʼʼʼʼʼʼʼʼʼʼ��g�d�]�^�g�q�s�����������������������s�g���y�I�I�V�A�Z�s���ʾ��������ʾ�����ƎƉƆƉƎƗƚƠƧưƳƵƳƬƧƚƎƎƎƎ�t�o�n�t����ùìââéìù���������������������Ҿ��о׾��������	�������	����׾ʾ¾¾��������׾��.�T���~���`�"������������������������
��#�8�/�
�����������������������)�B�N�P�S�P�H�B�6�)�����������������������$�0�=�<�1�/�$��¦¡¦²¿����������¿²¦¦¦¦����������������������������U�H�J�Q�U�a�g�n�p�x�r�n�e�a�U�U�U�U�U�U�L�9�3�'���'�3�@�L�e�r���������~�e�Y�L�Z�N�N�B�N�T�Z�g�h�g�c�e�Z�Z�Z�Z�Z�Z�Z�Z���	����������� �	��"�.�/�1�/�-�#���U�T�H�F�?�?�H�I�U�\�a�n�o�p�n�i�g�a�V�U������������������������輱���������ʼ����"�.�5�5�.�3�.���Ｑ�[�X�\�h�t�{āčĚĞĦĳĳĦĢčā�s�h�[����ݿĿ�����������Ŀݿ�����������@�<�<�@�L�Y�e�r�~���~�{�r�e�Y�L�@�@�@�@�?�4�)�'�!�&�*�/�.�4�@�M�S�Y�e�g�e�Y�M�?�������s�l�m�������*�F�S�j�W�-��ɺ�����äÓÄ�w�s�{�|Óù��������&�-������ŹŹŷŶůŹ��������������������������Ź�Ϲù����������ùϹܹ������ ������ܹϿ������������ �������������������������������������������������z�t�n�v�z�������������������������z�y�x�z�������������z�z�z�z�z�z�z�z�z�z�Y�W�N�M�Y�f�j�r�s�{�r�f�Y�Y�Y�Y�Y�Y�Y�Y�������������������������������������������s�v�����������������������������������b�]�V�R�I�D�I�V�]�b�o�u�{��{�o�b�b�b�b��н����n�e�i�u�����Ľݽ���@�6�1�#���f�M�U�Z�f�s�����������������������s�f�A�<�9�:�A�E�N�Z�g�s�s�v�u�s�i�g�Z�N�A�A���ܻܻڻܻݻ��������������黪�����������ûͻǻû�������������������������t�p���������������������������������������������(�)�5�?�A�5�)����������������������������������������������н��������������Ľݽ�����	�����������������(�.�,�(�������������������������������������"������������^�R�\�������������������������������*� �����������)�*�4�*�*�*�*�*�*������)�6�?�6�)�(����������{�r�s�}ŇœŠŭŹ������������ŹŠŔŇ�{�¡²¼������������¿²�ּּּۼ߼����� ���������ּּּּ���������������	������� �������x�r�l�_�S�R�L�l�x�������������������������������ĿѿҿݿݿѿĿ���������������EEEEE$E*E7EPE\EiEuE�E�EEuE\EPE7E*E�#�����������Ļ����������#�)�/�6�:�1�#ā�~�u�vāčĚĚĦīĲĦĚčāāāāāā�ù����¹ùʹϹйչϹùùùùùùùùùûû������������ûлֻлʻûûûûûûû� 3 5 @ = b : Y > > % q - Z M 8  ] . X M W L ( a � Q N V F K W [ 0 ! 3 2 W ; I I N { ; J $ $ g 1 X < G 8 l V ^ i A 9 ^ f p = f ? ; h d    0  =  �  2  �  #  X  h  �  �  E  �  �  �    �  E  �  q  �  B  \  i  `  �  �  �  �    �  �  N  w  �  J  y  �  ?  �    %  �  �  �  H  w  L  i  I  n  �  �  <  �  �  s    �      �  �  4  �  �  N  5��h�+��O߽P�`��t����-�ě��T���C���C��D���P�`��C����`�C��ixսm�h�+��1�+�\)�����o�t��ě������C���1�49X�T����1��+�49X��o��P��w�49X��P�0 Ž8Q�49X�0 Ž��ٽm�h�]/�T���#�
�q���T���L�ͽ�C��T���u�%�aG���%������罧����j���
�$ݽ�񪽸Q콩����B�B]TBH�B��B�B��B&�SA���B �A���BBBu�B��BܨB� B pBSB��BL�BS�B"��B3�A�bB!#�B ��B.<�B=�B�B		B�B��BIB�lBb,B�8B ��B�B�B �:BxB�KB
=�B�B>�BB��B:RBUNB	�rB/�B#�	B%�B��BG8B�?B�MBdkB
��B�B�hB �[B1LB��B	�}B
)�B�B'7B:uB��B �;B��B�B��B&�_A��,B ��A��BDBI�B�XB=~B��B AYB&,B�CB��BF�B#�B��A�nyB �B ��B.J�B?�B�mB��B�WB�B�B�BG	B��B�B�?B0!B ØB?�BB�B	��B��BYqB>B��B8�BENB	@B?�B#�'B%�HB� B�B�B�BB}aB
@B��BMB ��B��B�rB	�B
5�BŽB';�A��tAi?sA���@�$t@��]A;f1@�/A�R�AJ��BA�A�,�A�-AYhAZ
�A��A�,B	A�{�A�ǶA�f ?�:gA�naA�S�A� ,?�LA&�A���Ax��?�x�@��@=�AΟ A�2C>�ڒA�i#A�lmA��A���@��A�t�A���BF\A'a�AF�&A�Y�@���@�"BA�HDA��AJn]A(��A2`�A�q�A��NA�#�A���A�qUA�)�AJAn@�`�Ay��C��[A�A���>S��@�E�A�AmS�A�}@�#V@��VA;��@���A��AO6�BA}A�{A�t�AYn�AY	A�~�A�z�B	HPA��AA�`A�}?�АA��GA���A���?�AýAݍ,Aw<�?ڣH@��~@-6�A�|�A�>�|�A�wA�\A�
A��f@�:�A�}�A�Z�B�TA&�/AI�A���@�M�@�\A���A���AJ��A()�A3,�A�_YA��A���A�M)A�FFA�vA�>AI@��Ay��C��kA��hAހ�>}�}@��N   1   &   H   6      N         !         -   
   _      1   1      
                     +   *   8         7   t      !               
      	      U                     	            N                            8               #   5   -         /         5               9   )      -            %               1      )         9   5                                 7                        %         /                              %               '   %                  3                  )      )                           +               1   +                                 #               !         %         %                              %         O��yO�*7P6|O?��O �OU��Nq!O��PR81N���M�`�O�?nN��O��O頝O�'}P�tN���N'SBN3&fO>�N0{*N�N�_N�VO��O)s4O�k�N߭�OGB�P=K1P%�2O-�NO�N3N1N_ğO2&�N(PJN4��N��N���N�+�O�"<Olu9O�O0m|N��O���O�ONZ	�O��N���N�x�O���Ni+�NBwMO�O��zN���N�"WO�\N���Oj�O���N�KQM�+M�	    �  �  0  b    �  �    �  �    �  	!  �  �  x  D  7  �  �  �  �  �  �    	�  U  E  �  �  W  �  �  �  �  I  �  �  A  �  #  �  �  +  /    ?  �  4  �  m  ]  
�  �    �  �  L  �  	b  .  
�    �    <D����o��C��#�
�o�'�o�D�����
��`B���
�D���49X�]/�o��C��D���e`B�T����1��1���
�ě���j��9X��`B�+��w��h����w�]/�o�#�
�+�C��C��C��t��t��t���P�}����w�����#�
�',1�,1�,1�H�9��7L�H�9�Y��y�#�u�u�u��%������㽟�w���w�����������������������/6BO[ht{����th[B7/*/hsz�������������zpeh)6=BEHKKLJFB;6+>BO[chmjea[QSOJDB@?>
#+8<@A=<0/#��<<=IU`ZUI<<<<<<<<<<<HHTailmnmhaTRHFBABHHrv�����������������rEHTadmgaTHHEEEEEEEEE
#'&#
����
#+,)#
�������LN[[grtxtg[ONGLLLLLL��������������������)/<Han���zvUC)15/#)mqz������������znkkmms~��������������umm���������������������������������56BKOVWOBA7655555555�����������������|�����������||||||||FHLTabmprqmaTLHBFFFF�����������������������������������������������*:;6&����%)/55754-)
47;BFN[gt{~}{tg[NB84lnz�����zunidbbllll����),.%������������������������������)5DNYZQKD5)���)5>ABDB?5)	KOZ[_hsttvtth[SOGHFK����� �������������������������������������
#/30/*"
�����RUadlhaVULRRRRRRRRRR������������������������������������������������������������pt}���������ytnipppp��$143+"������),19@BEC@6)\abnz�������znmaa]\\����������������������������������������)6BFOTUOOB6)agkt�������ztg^a`_aa�������������������
)241!"
������!#0<BEED@<0+%#""!!!!���������������������)5BJNUOB5�����������������������������������������������������������������ghtw����������trgehg�����������������������#��������������������������TU[anwonaULOTTTTTTTT��
"$! " 
������MO[_gkt���������t[NMqt��������tpjjqqqqqqMUanongaUOMMMMMMMMMMGIPSUVZ^ZULIGGGGGGGG�����������
��#�/�<�H�U�Y�\�W�H�<�/�#���.�)�$�'�/�.�8�G�`�m�������������m�T�;�.��ĿĻĲĴĺļ�����������	����	�����̻S�F�.�)�-�3�:�F�S�_�l�x���������x�l�`�S�������������ûлܻ����ܻػлû��������(�!�� �(�4�;�A�M�Z�[�f�g�h�d�Z�M�A�4�(�ʼ����������ʼϼѼʼʼʼʼʼʼʼʼʼʼ��g�f�^�_�g�s�{���������������������s�g�g�������S�N�X�K�I�Z�s���ʾ����
�	��ʾ�ƎƊƈƌƎƚƧƭƲƧƦƚƎƎƎƎƎƎƎƎ�t�o�n�tùìæäååîù����������������������ù��������� �	����	������������׾̾ɾԾ����	��"�.�8�D�3�"������������������������
��#�8�/�
�����������������������)�6�B�I�K�N�K�B�6�)����������������� ��$�0�<�;�3�0�-�$��¦¦²¿����¿²¦¦¦¦¦¦¦¦¦¦�� ����������������	���������U�T�O�U�W�a�n�s�o�n�a�W�U�U�U�U�U�U�U�U�L�@�@�G�L�Y�e�r�~�������������~�r�e�Y�L�Z�N�N�B�N�T�Z�g�h�g�c�e�Z�Z�Z�Z�Z�Z�Z�Z�	�����������	��"�*�/�)�"���	�	�	�	�H�F�@�@�H�K�U�]�a�n�o�p�n�h�f�b�a�U�H�H��������������������������ּʼ�������������!�&�.�2�2�&�����h�]�`�h�l�tāĂčĒĚĦĭįĦěčā�t�h�ݿĿ����������������Ŀѿݿ�������ݺ@�<�<�@�L�Y�e�r�~���~�{�r�e�Y�L�@�@�@�@�2�'�&�'�(�,�1�1�4�@�M�R�Y�d�f�c�Y�M�@�2�������}�w�z�����ֺ���0�;�9�!���ֺ����������ïÃ�ÇÑìù��������������ŹŹŷŶůŹ��������������������������Ź�ù����������ùϹչܹ�����������ܹϹÿ������������ �������������������������������������������������z�t�n�v�z�������������������������z�y�x�z�������������z�z�z�z�z�z�z�z�z�z�f�Z�Y�P�R�Y�f�h�p�r�t�r�f�f�f�f�f�f�f�f�������������������������������������������s�v�����������������������������������b�]�V�R�I�D�I�V�]�b�o�u�{��{�o�b�b�b�b�����|�~�����Ľнݽ���������ݽн����s�i�O�W�Z�f�s�����������������������s�N�B�A�;�<�A�G�N�Z�g�q�s�u�t�s�h�g�Z�N�N���ܻܻڻܻݻ��������������黪�����������ûͻǻû��������������������������u�q�|�������������������������������������������(�)�5�?�A�5�)����������������������������������������������н��������������Ľݽ�����	�����������������(�.�,�(�������������������������������������"����������s�c�i�w�������������������������������*� �����������)�*�4�*�*�*�*�*�*������)�6�?�6�)�(���������ŔœŇŇŊŔŠŦŭŹŻ����������ŹŭŠŔ¥¦²¹¿����������¿²¥�ּּּۼ߼����� ���������ּּּּ���������������	������� ���������x�s�l�a�_�T�]�l�x�����������������������������ĿѿҿݿݿѿĿ���������������EEEEE$E*E7EPE\EiEuE�E�EEuE\EPE7E*E�#�����������Ļ����������#�)�/�6�:�1�#ā�~�u�vāčĚĚĦīĲĦĚčāāāāāā�ù����¹ùʹϹйչϹùùùùùùùùùûû������������ûлֻлʻûûûûûûû� : G : ? a + Y * ? / q & P @ 8  a . [ V ? L % c � G G - F M B S 0  3 2 W ; G I N { 2 J  $ g 4 X < G 8 l < ^ i 5 2 ^ f m = f ? ; h d    �  [  a  �  >  �  X  %  �  �  E  S  �  Y    1    �  `  e  �  \  �  *  �  _  s  �    �  c  =  w  >  J  y  �  ?  \  �  %  �  F  �  (  w  L  H  I  n  �  �  <  -  �  s  @        R  �  4  �  �  N  5  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  F
  �  �                �  �  �  	  �  �  E  �  �  �  z  l  �  �  �  �  �  �  �  �  �  [  �  �  �  �      �  l  �    j  �  �  �  �  �  �  �  �  �  x  2  �  ^  �    K  m  |  �  �    #  0  '    �  �  x  F    �  T  �  i  �    L  �    -  J  ^  _  U  M  ?  ,    �  �  �  �  `    �  |  5  �    f  �  �  '  T  �  �  �        �  �  t  �  *  &  �    �  �  �  �  �  �  �  �  |  r  h  ^  a  l  w  �  }  s  i  _  U  n  �  �  y  k  ]  M  <  *    �  �  �  �  x  R  +    �  �  �        �  �  �  �  |  F    �  �  s  5    �  �  H  s  ~  �  �  �  �  ~  v  l  ^  >    �  t  7  �  �  b    �  �      .  9  C  O  ]  k  |  �  �  �  �  �  )  z  �  �  �  �  
            �  �  �  8  �  �  D  �  �  '  �  t  j  <  W  r  }  �  �  �  �    s  e  V  F  5  #        =  _  C  �  `  �  
  g  �  �  	   	  	  �  �  U  �  *  O  g  t  �  �  v  [  /  �  �    4  �  �  _  ,    �  �  �  �  �  �  ~    f  �  �  �  s  M    �  �  c    �  8  �  )  o  �  �  <  L  v  e  E    �  �  V    �  H  �  �  �  R  �  T  �  �  �  /  &  6  D  4      �  �  �  �  {  c  J  &  �  �  �  �  �  4  5  7  7  7  4  1  -  '         �  �  �  w  l  a  W  L  %  O  s  �  �  �  �  �      �  �  �  �  M  �  �  A  �  m  #  W  �  �  �  �  �  �  o  V  8  *  >  ?  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  g  V  D  2       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  T  9    �  e   �  �  �  �  �  �  �  j  B    �  �  �  �  �  �  s  U  D  =  <  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �             �  �  �  a  %  �  �  �  s  S    �  0  !  	G  	�  	�  	�  	�  	|  	V  	)  �  �  o    �  "  �  �  /  x  �    i    D  T  S  J  7    �  �  p    �  a  �  �    x  �  �  E  >  6  )    �  �  �  �  �  e  ;  
  �  �  H  �  }  �  b  �  �  �  �  �  �  �  �  �  �  u  L    �  �  �  O  �  \  r  g  z  �  �  �  �  r  M  &  �  �  �  8  �  V  �  =  �  �  W  (  �    N  T  B  '    �  h  �  g  
�  
  	F  V  /  �      �  �  �  l  L  +    �  �  �  �  �  e  C    �  y    �  �  0  S  k  �  �  �  �  �  o  Q  /  �  �  x  D    �  F  �  �  �  �  �  �  �  �  �  w  k  ^  P  ?  /      �  �  �  �  �  �  �  �  �  �  �  �  �  �  f  K  0    �  �  �  �  o  I  "  I  8  '    �  �  �  �  �  u  ^  N  ;  %    �  �  �  ]  /  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  h  \  O  �  �  �  �  �  �  �  �  �  �    z  s  n  j  g  e  f  r    ?  @  A  =  8  /  &        �  �  �  �  �  �  y  &  �  ]  �  �  s  g  Z  N  C  7  )      �  �  �  �  �  �  t  Z  ?  #    
  �  �      �  �  �  �  �  �      �  {  [  <    �    ?  _  ~  �  �  �  u  O  '    �  �  w    �  �    S  �  �  �  �  �  �  d  F  +    �  �  �  �  N    �  �  �  Z  "  )  %    
  �  �  �  �  y  T  .    �  �  l  +  �  �  #  /  ,  '        �  �  �  �  �  �  �  �  m  V  =  (  �  J      �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  n  `  R  5  ?  8  %    �  �  �  �  �  a  /  �  �  e  �    )   �   �  �  �  �  �  �  �  �  s  [  G  9  !        �  �  �  �  �  4  /  )  #        �  �  �  �  ~  Q  0    �  �  �  t  K  �  �  �  |  f  V  I  4      �  �  �  �  o  3  �  �  *  �  m  j  f  X  I  9  (    �  �  �  �  |  Y  7    �  �  �  �  ]  Z  U  L  L  S  @  (    �  �  �  o  *  �  �  +  �  w  0  
/  
�  
�  
�  
�  
�  
�  
{  
M  
.  	�  	q  	   �  �  m  �  �  �  f  �  �  �  �  }  v  n  \  E  .      �  �  �  �  �  �  �  �      �  �  �  �  �  �  �  �  v  d  R  @  #    �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  F    �  �  �  �  �  �  �  �  �  n  I  "  �  �  t  c  R  A    �  F  �  5   �  L  $  �  �  �    V  -  �  �  �  5  �  j  �  '  �  �  `  �  �  �  �  �  �  �  �  �  v  V  6    �  �  �  v  L  ?  E  ]  	  	\  	I  	  �  �  f  $  �  �  0  �  q  /    �  �  0  �    .         �  �  �  �  �  �  t  ^  G  0    �  �    E  �  
�  
�  
d  
I  
+  
+  
x  
�  
�  
�  
x  
O  
"  	�  	�  	s  	1  �  �  O      �  �  �  �  �  `  5    �  �  �  b  &  �  �  �  a  �  �  �  �  �  �  o  R  /    �  �  �  c  9    �  �  �  F  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �         "  #  !        �  �  �  �  �  �  �  x  d  P  <