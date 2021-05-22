CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��l�C��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�S   max       Ppv�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��^5   max       <�o       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�
=p��   max       @Fg�z�H       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @v�(�\       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P            �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @�           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �!��   max       <D��       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B1 '       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�t   max       B0��       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�<�   max       C��=       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C���       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          n       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�S   max       P;P       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Q�`   max       ?��u��!�       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���`   max       <�o       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��Q�   max       @Fg�z�H       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�(�\       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q@           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�K        max       @���           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Es   max         Es       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?��҈�p;     0  ^                  	            
      	         
               '                              ^   8                     <               M               "      m   !   .            /               
         D                  <    N��ZNJ<�N�N
�OG�1O	yN���N��COj|�N���N�}N��N��XN�a�O��O�dM��OP�IO��>O�ɭM蒂NǌO�@�O�nM��O-�6N��RO���O�w�O���P�>O���O灘OQ�N�͢N�QsN���O�nN�,O/��N�T�OUPB]`O<�mN�a�N��@N!ORf�Nx(�Ppv�On�^Oٌ�N$�OJ��N�#=O�U�M��9O�b�O<�N�C�Ofi�N�|OS:�P�N��*O2?�N�}�M�SO��#OqLO�UX<�o<D��<o;ě�:�o%   ��o��o���
���
���
�ě���`B�t��#�
�T���u��C����㼣�
���
��1��1��j��j��j�ě����ͼ���������`B���+�C��C��C���P���������''49X�8Q�<j�<j�@��D���H�9�L�ͽL�ͽe`B�e`B�ixսixսm�h�u�u��%��C���C���C���hs��������������^5��^5��^5��������������������jmrzz{zusmihhijjjjjjOUanvnhaUJOOOOOOOOOO
)+)#







��������������������������������������������������������������������������������agnrxz}�������zla_]a
#(,#
v{�����{qpvvvvvvvvvv��������������������"#/;<HPMHC=<://# ""aaimz���zsmia__aaaacotsz�����������zn`c�������������~���������������������������������������������"*6CO\n|�o\O6*"������#.61)$
����#+/1/*##" �����������xtptty���������	���������������������������������������������������������������������������������������������)6@;)������0BCM[gt��������tNDB0����
#,/:8*�������������

��������������	
��������#1Ien{|gb_VI<0#
)5BNY[agc[B5)"����������������������������������������8BO[^hjh\[OJB7888888����#)1)"��������')5:BNPRNBB5,)!''''TUZ]ajnz�����znaUMRT>BEQ[cpt{vtljihdf[O>`hkmz�����}zpma\ab``������*-&�����������������������������������������������t�����������������tt�������������>CKOS[hlt~�unh[ODB<>��������������������CFQt�����������t[G@C����������������������������������������qz��������ztqqqqqqqq��������

������������������������������������� �����������������������������qt���������������tqq��������������������\aemxz�����}zrma\W\\��������������������*/2<HTUXVUH</'******�������		���������)5KN[^eklqle[NB.%")#$*./262//#!=HUYahiiheaXUPJHA@>=##0<=?<;8740+(%#  ##/09<CB?<0.//////////���������������������������������������������	�������¿¹¿��������������������������¿¿¿¿������������)�1�)��������������������������������������������������������a�Z�T�P�I�T�Z�a�e�m�o�m�a�a�a�a�a�a�a�a�<�3�/�,�$�#�"�&�,�/�<�H�Q�U�X�Z�U�O�H�<�s�r�f�e�_�f�s�|���������������������s�H�B�<�6�5�<�H�U�a�b�b�a�_�U�H�H�H�H�H�H�U�I�H�F�H�T�U�\�a�g�n�s�zÁ�z�n�f�a�U�U�H�;�3�/�"� �"�/�;�T�a�k�m�~���z�k�a�T�H���������������üʼ˼мʼ����������������Ľ��Ľǽнݽ޽�ݽнĽĽĽĽĽĽĽĽĽĻ!����!�-�:�F�S�S�_�e�_�X�S�F�:�-�!�!�)�)�(�)�)�.�6�B�J�O�[�]�[�O�O�G�B�6�)�)ÓÓÇÁ�{ÇÓàìöùûùñìàÓÓÓÓ�������������������������#�-�)��������������������������������������f�c�Z�Y�X�Y�^�f�r�s�|�r�f�f�f�f�f�f�f�f�������������żռּ��������ּʼ���	���������	��"�.�5�7�6�3�.�)�"�����ìÍÇ�{�|�{Çàù�������������������H�E�H�N�U�]�a�n�t�n�a�U�H�H�H�H�H�H�H�H¦¦©§²º¿��������������¿½²¦¦¦àÓË�z�y�|ÇÓÙàæìùü������ùìà�$�������$�(�0�8�=�?�A�A�=�3�0�$�$�����������������������������������������������������������������ǾʾоҾҾ̾ʾ��G�E�;�8�8�;�<�G�T�]�`�a�`�`�T�I�G�G�G�G���z�n�h�i�n�{���������������������������	�������������	��!�+�/�,����	ED�D�D�D�D�EEEE%E*E7EPEWEUEOECE7EE�����~�x�y�������Ŀѿ���������ѿ��������׾˾žʾԾ�	��.�5�G�W�T�;�.�"�����������y�g�\�^�s�����������������������=�0�-�,�/�,�-�0�2�=�I�V�Z�Z�Z�X�V�R�I�=�������ݿԿݿ�������������������������������������Ŀ̿Ŀ����������������������������ûʻллӻлȻû������������[�Y�Z�c�nĔĚĥĦĨĳļļķĦĚč�t�h�[�����������������������������������������Z�W�N�A�:�A�D�Q�W�Z�g�s�~�~���}�w�s�g�ZìçàÓÑÓàæìùú��������������öì�ݿѿĿ����Ŀѿݿ߿��������������ݼf�Y�W�w�����ʼּ����������ּ���f�@�4�.�'�*�4�@�M�Y�f�r������z�r�g�Y�M�@�C�9�C�O�O�\�h�u�|�u�m�h�\�O�C�C�C�C�C�C��������������������
�����	�������H�@�;�/�.�/�;�F�H�I�T�W�T�Q�H�H�H�H�H�H�Ϲù��������������ùϹܹ������ܹ���������)�*�-�)�&���������h�[�O�?�;�B�O�[�tčĚĦĳ��������ĳč�h�;�1�"��"�/�;�H�T�a�m�r�v�v�m�k�a�T�H�;���~������������������������������������������������� ����������������������������������
��$�1�5�5�0�)�#����<�:�0�#��#�.�0�<�I�U�b�l�j�b�U�L�I�<�<��ĿĳĲİĲĿ����������������
�����ػ����������������������������������������<�3�+�,�)��#�0�<�b�{Ňőł�{�n�e�Q�I�<ŠŗŘŕŞşŦűŹ������������ŹŰŦŪŠ��ƸƳƭƧƥƣƧưƳ���������������������s�g�n�s�������������������������������s�����������������ʾ˾ѾվӾʾ������������Y�X�L�@�.�%�$�'�3�@�L�\�e�o�u�u�r�k�e�Y�a�J�/�&�$�/�<�U�zÓìù������ùìÓ�z�aE�E�E�E�E�E�E�E�FFFFFFE�E�E�E�E�E�FFFF$F0FJFVFcFoF|F�F|F{FoFcF]F=F1F$F�������������������ĽнܽнĽ������������f�d�Z�U�Z�f�s�w�w�s�f�f�f�f�f�f�f�f�f�f�������������ɺֺ����*�!�����ֺ��k�_�S�F�:�:�F�J�_�x�����������������x�k�!�������!�.�S�`�l�s�j�Z�R�G�:�.�! P � p t J + 2 F , %   @ < = Y L k ; K B l / I 3 ` @ 6 [ S ; 8 [ r f ] a e O ? < X Y a ? n N 6 * ] # 1 D n q k ' c F | ] ] : O J d p l Q X = 0    �  �  ^  S  �  8  �  �  �  �  *     �  �  0  `  2  �  �  $  F  �  D  2  /  �  �  7  �    �    �  �  �  �  �  �  �  �  `  G  �  �  �  1  <  �  �  �  �    �  '    0  !  �        �  �  �  �  �  �    }  �  ?<D��;�`B;D��:�o��j�t���o��t��u�e`B�D���e`B������ͼ��
������''q�����ͼ�h�H�9�,1���ͽ\)���8Q�e`B��F����m�h�P�`�L�ͽ�w��P�L�ͽƧ�0 Ž8Q�49X�<j�������P�`�L�ͽY����
�]/�!����\�u��t���O߽�񪽋C����㽕����-���w���
����V��1��j��-���-���پ�����#Bm�A��B'4B�0BB�B!>�B��BY�B$�B(־B,:�B�OA��iB�;B}ZB @�B ϝB1 'B��B��B�BY7B"�B7OB!�<B �B�>B	��B[�B�BƶB&QEBv�B*&�B+W�B�PB��B]_B��Bd�A��B-?GB�B_�B(B7�BK�B�B
LB��B&VB WBr`BާB��B �SB\B�OA���BCB�B"��B6!B�B��B%��B&$�B3�Bf�BYB?�A�tBǸBG�BܶB>�B!=aB�HBuuB$�iB(�-B,K�BJ@A�tB �B��B <'B �:B0��B��B��B��B��BB?�B!�?B 9EBH%B	֤BJB�#B��B&A�B�MB*�B+��B�hB��B?�B��BGcA���B-��B9�B��B7�B>�B@�B=,B	B�BE�A���BC�B��B��B B�B6(B@A�{�B�B�OB"�BŧB��B�}B%�xB&;LBN�B?�B��A��}A�fA��?A��A��AE�}A���A�*�A�Te@��
A)u�@}�nA�uA�B>A�*rA��@��VA �tA]m�A͚A��(A��BA�l�B
 �A��AM�Ae}�A��8A[udC�}hAx|�AZ�TA�,KB
�QA�-�At5'@�&�A�i0A��[A�9�A�H�A}5$@�ea@�X*B�fA�´A�#>�<�A�3�A�^�A�pA���A�fgB	>yA��_A�<f@��A�+A��eB�A�cAO�?��2A�߸C���C��=A#��AAD�@B��@��&Aa'A���AӸ?A�\VA��DAÕ/AE�nA�[IAƁ5A��@��eA)�@xܶA׆�Aʄ�A���A��}@�]@��@A[VA�yYA�vA�saAˣyB	�,A�|;AM��Ae'*A�ρA[!C��Ay5�A\ڵA���B
�LA�Y�AuPc@��5A��A��A�&�A�yNA}�+A�@�=�B>�A�H@A�ni>��A��{AۇIA�r~A���A�B	�A�W�A�l,@��'A�<A�yBZA���AP�W?�#AȀ�C�xkC���A"ɿAAL:@D�@�(�A��                  
            
      	                        (      	                        ^   8                     <               N               "      n   !   .            /                        E                  =                                                 !               #                        !      #   '   #   +                              7                     3      #                                    %               !                                                                  #                        !         #      +                              3                           !                                    #               !      N��ZN�nN�N
�OG�1O	yNZ��M�wOj|�Nv
8N�}N��N��XN�a�Om��O�dM��OP�IO��>O�ɭM蒂N�	�Oz��N�+wM��N�X�N��RO���O=/O���O�ޚO��gO灘OQ�Nk�FN�QsN���Oz*JN�,O/��N�w�N�L�P;PO-��N�a�N��@N!N��uNx(�O��OK��O��N$�OJ��N�w�O��M��9O�b�N�޵N�'jOfi�N�|OS:�O�CN��*O2?�N�}�M�SO��POqLO�UX  �  B  �  �  x  �  �  $  W    �  �  �  �  �  �  D  �  �  �    �    R    �  �  a  /  �  y    �  �  	  N  9  
�  �  �  �  �  	t  �  5    �  �  �  Z  �  �  �  f  �  �  �  �  �  5  �  H  �  
b    �  |  Z  �    S<�o<49X<o;ě�:�o%   ��o�#�
���
��`B���
�ě���`B�t��49X�T���u��C����㼣�
���
��9X��j��/��j��/�ě����ͽ\)�P�`�\)�t��+�C��\)�C���P�@�������w�,1�0 Ž8Q�8Q�<j�<j�}�D�����`�Y��P�`�e`B�e`B�u��t��m�h�u��%�����C���C���C����
��������������j��^5��^5��������������������kmpyzzzspmkjiikkkkkkOUanvnhaUJOOOOOOOOOO
)+)#







��������������������������������������������������������������������������������agnrxz}�������zla_]a
#%'#

v{�����{qpvvvvvvvvvv��������������������"#/;<HPMHC=<://# ""aaimz���zsmia__aaaaeqvu}�����������nece�������������~���������������������������������������������"*6CO\n|�o\O6*"������#.61)$
����#+/1/*##" qtv{��������{tqqqqqq����������������������������������������������������������������������������������������������������)6@;)������LNR[gt���������t[TNL����
%&$
��������������������������������
�����#1Ien{|gb_VI<0#
)5BNY[agc[B5)"����������������������������������������8BO[^hjh\[OJB7888888������������')5:BNPRNBB5,)!''''TUZ]ajnz�����znaUMRTHOS[ahntytrh[XONHHHHlmz~���{zmebajllllll�����*,%������������������������������������������������t�����������������tt�������������KOY[chsqhg[OKGKKKKKK��������������������Ygt����������tg[WTUY����������������������������������������qz��������ztqqqqqqqq��������

�������������������������������������������������������������������qt���������������tqq��������������������^ahmz�|zqma^Z^^^^^^��������������������*/2<HTUXVUH</'******�������		���������(-5N[dgini`[NB5*%"#(#$*./262//#!=HUYahiiheaXUPJHA@>=##0<=?<;8740+(%#  ##/09<CB?<0.//////////���������������������������������������������	�������¿¹¿��������������������������¿¿¿¿������������)�-�)��������������������������������������������������������a�Z�T�P�I�T�Z�a�e�m�o�m�a�a�a�a�a�a�a�a�<�3�/�,�$�#�"�&�,�/�<�H�Q�U�X�Z�U�O�H�<�s�r�f�e�_�f�s�|���������������������s�H�D�<�:�<�=�H�U�^�_�\�U�H�H�H�H�H�H�H�H�a�Z�Z�_�a�c�n�o�z�{�z�n�a�a�a�a�a�a�a�a�H�;�3�/�"� �"�/�;�T�a�k�m�~���z�k�a�T�H�����������������ȼʼʼʼ����������������Ľ��Ľǽнݽ޽�ݽнĽĽĽĽĽĽĽĽĽĻ!����!�-�:�F�S�S�_�e�_�X�S�F�:�-�!�!�)�)�(�)�)�.�6�B�J�O�[�]�[�O�O�G�B�6�)�)ÓÓÇÁ�{ÇÓàìöùûùñìàÓÓÓÓ�������������������������,�)���������������������������������������f�c�Z�Y�X�Y�^�f�r�s�|�r�f�f�f�f�f�f�f�f�������������żռּ��������ּʼ���	���������	��"�.�5�7�6�3�.�)�"�����ìÍÇ�{�|�{Çàù�������������������H�E�H�N�U�]�a�n�t�n�a�U�H�H�H�H�H�H�H�H����¿²©«²¿������������������������àÚÓÌÄ�{�z�~ÇÓàâìù����ÿùìà�$�#������$�0�<�=�?�?�=�0�'�$�$�$�$�������������������������������������������������������������ɾʾ˾ʾǾ����������G�E�;�8�8�;�<�G�T�]�`�a�`�`�T�I�G�G�G�G���z�n�h�i�n�{���������������������������	����������������	�� �%�$����	D�D�D�D�D�EEEE*E7ECEPEQEMEEE7E*EED����������������Ŀѿ��������޿ѿ��������	�����׾Ӿ;׾���	�"�.�8�@�:�.����������y�g�\�^�s�����������������������=�0�-�,�/�,�-�0�2�=�I�V�Z�Z�Z�X�V�R�I�=������������������������������������������������Ŀ̿Ŀ����������������������������ûʻллӻлȻû������������h�`�`�h�uĈčĚĥĦĳĸĸĳĦĚčā�t�h�����������������������������������������Z�W�N�A�:�A�D�Q�W�Z�g�s�~�~���}�w�s�g�ZàØÓÑÓÚàèìõù��þùììàààà�Ŀ¿��Ŀѿݿ���������ݿѿĿĿĿĿĿļf�Y�y�����ʼּ�����������ּ���f�@�4�.�(�+�4�@�M�Y�f�o�r�}�x�r�f�e�Y�M�@�C�9�C�O�O�\�h�u�|�u�m�h�\�O�C�C�C�C�C�C��������������������
�����	�������H�@�;�/�.�/�;�F�H�I�T�W�T�Q�H�H�H�H�H�H�ù����������ùϹ۹ܹ��ܹϹùùùùù���������)�*�-�)�&���������[�R�L�J�K�O�U�[�h�tāčĝĢěĉā�t�h�[�;�4�&�#�/�6�;�H�T�a�m�p�t�t�m�g�a�T�H�;����������������������������������������������������� ����������������������������������
��$�1�5�5�0�)�#����<�0�<�@�I�U�b�h�f�b�U�I�<�<�<�<�<�<�<�<�������������������������� �����������̻����������������������������������������<�3�+�,�)��#�0�<�b�{Ňőł�{�n�e�Q�I�<ŭťţũŭŵŹ��������������ŽŹŭŭŭŭ��ƻƳƯƩƳ�����������������������������s�g�n�s�������������������������������s�����������������ʾ˾ѾվӾʾ������������Y�X�L�@�.�%�$�'�3�@�L�\�e�o�u�u�r�k�e�Y�n�a�Q�6�8�H�U�zÓìùýÿþøìàÓÇ�nE�E�E�E�E�E�E�E�FFFFFFE�E�E�E�E�E�FFFF$F0FJFVFcFoF|F�F|F{FoFcF]F=F1F$F�������������������ĽнܽнĽ������������f�d�Z�U�Z�f�s�w�w�s�f�f�f�f�f�f�f�f�f�f�������������˺ֺ������!����⺿���k�_�S�F�:�:�F�J�_�x�����������������x�k�!�������!�.�S�`�l�s�j�Z�R�G�:�.�! P � p t J + + c , "   @ < = _ L k ; K B l ( F + ` 3 6 [ I : 9 a r f P a e F ? < A K ^ 5 n N 6 - ]  & B n q U  c F Q M ] : O @ d p l Q R = 0    �  �  ^  S  �  8  m  B  �  �  *     �  �    `  2  �  �  $  F  �    �  /  �  �  7  �  :  �  4  �  �  h  �  �    �  �  �  �  �  q  �  1  <  �  �  1  �  �  �  '  �  H  !  �    �    �  �  0  �  �  �    H  �  ?  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  Es  �  �  �  �  �  ~  v  n  f  ^  T  F  8  +              ?  @  @  A  B  C  D  F  N  `  r  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  h  U  A  +    �  �  �  �  \  2    �  �  S    �  f  �  �  �  �  �  �  �  �  �  �  }  j  V  ?  %    �  �  �  [  (  �  �  �  �  �  �  �  �  �  l  W  B  ,    �  �  �  �  f  �        �  �  �  �    !  �  �  �  �  �  �  U  �  �    �  W  R  M  F  ?  9  5  9  6  *      �  �  �  �  �  �  C   �                    �  �  �  �  �  �  �  c  *  �  �  �  �  �  �  �  �  �  �  �  �  z  k  Z  H  6  $    �  �  �  �  �  �  �  �  �  �  �  v  i  ]  Q  F  >  6  /  (        �  �  �  �  �  �  y  [  <    �  �  �  a  1    �  �  �  �  �  �  �  �  s  _  B  !  �  �  �  �  e  ;    �  �  ?  X  �  �  �  �  �  �  �  �  �  �  o  \  R  B  /      �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  h  H  "  �  �  �  �  �  D  B  @  >  <  6  %      �  �  �  �  �  �  �  �  �  z  l  �  �  �  �  �  �  �  �  x  p  a  K  (  �  �  |  A    �  �  �  �  �  �  �  x  j  X  E  1      �  �  �  t  9  �  �  �  �  �  �  k  I  0  "    �  �  �  �  C  �  �  
  �    h  !              �  �  �  �  �  �  e  :    �  �  �  Q    �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  N     �   �   �  �         �  �  �  m  9    �  �  I  �  �  3  �  �  �    =  D  J  Q  R  <    �  �  �  a    �  �  N    �  r    �       �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   {   s   k  �  �  �  �  �  �  �  �  �  }  x  t  l  a  S  B  1       �  �  �  �  �  �  �  �  ~  z  u  n  e  [  N  :  &     �   �   �  a  _  W  I  4    �  �  �  ~  t  d  h  o  a  D  "  �  �  W  �  �  �    !  ,  -  #    �  �  �  �  b  0  �  o  �  3  �  �  <  l  �  �  �  �  �  ?  �    )  
�  
�  
>  	�  �  �  ?  :  P  o  w  x  w  t  n  _  A    �  �  d    �  q     �  �    �  �  �        �  �  �  n  5  �  �  �  }  5  �  d  �  �  �  �  �  �  �  �  �  �  s  \  D  ,    �  �  �  �  q  :   �  �  �  �  u  e  T  @  (  
  �  �  �  �  g  ?        �  �  �  �         �  �  �  �  �  �  �  �  �  k  T  ;        �  N  E  <  3  *  !                        �   �   �   �  9  8  3  )      �  �  �  �  l  I  $  �  �  �  �  V  )  �  
1  
�  
�  
�  
�  
�  
�  
�  
�  
N  
  	�  	�  	5  �  _  �    �  Y  �  �  �  �  �  �  �  �  �  �  �  �  }  v  o  h  a  [  T  M  �  �  �  �  �  �  {  i  W  E  2    	  �  �  �  �  �  s  S  �  �  �  �  �  �  �  �  s  e  V  H  :  4  <  D  B    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  p  Z  A  (    	o  	o  	\  	_  	6  	  �  �  C  �  �  @  �  �  �  e  �    �  �  �  �  �  �  w  h  W  @  &  �  �  �  H  �  �  <  �  u  w  �  5  !    �  �  �  �  �  �  �  �  t  f  [  T  M  E  3  !              �  �  �  �  �  �  �  �  �  �  �  R  "   �   �  �  �  �  �  �  {  f  S  @  +    �  �  �  �  �  v  Y  <    �  +  Y  x  �  �  �  �  �  �  �  �  �  _  2    �  �  ;  �  �  �  �  �  �  �  �  �  �  �  s  i  ^  N  1    �  �  �  �  	�  
�    H  R  <  $  )  C  Y  B  
�  
�  
  	P  �  �  �  �  8  }  �  �  �  �  �  �  �  p  R  0  �  �  �  T    �  r    t  �  �  y  j  Z  ?    �  �  �  d     �  s    �    q  �  l  �  �  �  �  �  �  �  �    w  i  U  B  .            �  f  T  B  .      �  �  �  �  l  ;    �  �  �  �  z  A  �  �  }  s  j  �  �  x  _  E  ,    �  �  �  �  �  �  �    W  W    �  �  �  �  �  �  �  �  v  G    �  1  �    d  O    �  ?  �  o  ^  N  @  6  ,  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  R  2    �  �  m  @    �  D  �  �  �  �  �  �  �  �  �  }  _  C    �  �  w  I  M  =    )      ,    �  �  �  u  R  /    �  �  c  '  �  �  _    �  �  �  �  �  �  �  �  �  �  �  �  }  r  f  Z  S  M    �  H  5       �  �  �  �  {  ]  =    �  �  �  J  �  �  R  �  �  �  �  �  �  x  Z  <      �  �  �  �  �  R    �  �  a  
  
U  
`  
a  
N  
-  
  	�  	�  	*  �  u    �  &  �  �  �  �  �       �  �  �  �  W  ,    �  �  j  /  �  �  �    Y  e  �  �  v  d  N  4    �  �  �  �  z  S    �  �  /  �  [  �  T  |  b  D  $     �  �  �  �  �  �  V    �  �  P  )        Z  S  K  D  =  5  -  $        �  �  �  �  u  P  +    �  �  �  �  x  Q  %  �  �  �  a  "  �  �  Q  �  �  N  �  v       
�  
�  
�  
�  
�  
Y  
  	�  	�  	K  �  �  	  j  �  �  �  �  �  S  @  *    �  �  �  N    �  w  1  �  �  Z    �  "  m  