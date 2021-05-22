CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�Q��R        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�   max       P�S�        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �Ƨ�   max       <�j        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>c�
=p�   max       @F'�z�H     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ҏ\(��    max       @v�
=p��     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q@           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�ڀ            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <�t�        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/3m        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�`�   max       B.�_        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C��i        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          `        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          G        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�   max       P�        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?զ�'�        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �Ƨ�   max       <�j        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>c�
=p�   max       @F��
=q     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ҏ\(��    max       @v�=p��
     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?ե��u�        W�         "                     _   	      	            	            )         
            W                  3      M   -               1         Q   U         4                  $      /      K               N���O�͍O0nNAp7O�?RO��O�3O�rNw��P��HNuRrOB�1OfKNggN�`O'�DNɩ�Ok?�N$�O��>Oݟ�M�O!(LNRliOћ�P+�N���PhrN�@�O?\bO�7LN���OF̨PE[�M��PN�EPB�O��O��N���N�!�O˧8O$��N,V�P3`)P�S�O�(�O"�}P8njNb�=NJ�LN��O,�QN�N�O���O+��O� �N rO��0OT�PO��<N��N���O-�<�j:�o%   ��o�o�D�����
���
��`B�o�o�t��49X�49X�D���T���T���e`B�u�u��C���t����㼛�㼣�
��1��9X�ě��ě��ě���/��/�����������o�o�+�+�C��C��t��t���P������w��w�8Q�<j�@��D���D���L�ͽP�`�T���]/�e`B��E���Q콸Q�Ƨ�mt{����������ytjmmmm"5BN[]^[SNB5)
	!)+00,)'��������������������#
����������� (+'#�������������������������������������������������������������������������������[am�����
	���zma\[��������������������lnqz{������zmfildjl���		 ��������:BHOPQOB@6::::::::::,/::<>HINUa^UH</.-,,Q[ct������zuslh[XRQ��������������������)06BObhmqwwtj[OB6)')zz���������zyxzzzzzz��*6=@CHOOC6*����U[gt����������tgbWSU������������������������� �����������������������������9B[ht�������t[GB<769;Haz����zhYMH;31523;=BNX[b^[NB97========��obI<3.(0IU`q{���*/9;<EF@;/"$********46BOX[^a[WOB660-*+.4������������������������	�����������������������������������25/�������!#/31/&# !!!!!!!!!!#/HYn�����zaUH<+##��������������������������������������������������������)-54)6BOW[cda\[XOIB:66666������������������������������������~}~�!#,/0552/%# !!!!!!!!����
(+&)24'��������� 0'(>GI$������0;DNt���������t\NB40IUbn{�����{nb_QKIBFImqz���������������vm)/0.+) ��������������������?BMOU[\[OJB6????????��������������������� ����������'Nt���������tg[B5!'������� �����������������������Y[cghkmjg\[ZYYYYYYYY����#'.2GMH<#
�������������������������
#/<BEHLOJH<# 

����������������������������������������)-5CN[]WSOMHGDB950))���������������������������N�K�9�5�3�3�4�A�Z�g�t���������x�g�g�Z�N�ֺɺ��������Ǻɺֺ���������	����ּ�����r�p�r������������������������������ʾϾȾ������������{�q�n�s�z����������(����%�,�A�c�s�}���������l�Z�M�4�(�A�9�4�4�*�&�(�A�N�Z�g�s�����������s�g�A���ݽν����������нڽ�����������U�S�H�E�H�U�a�n�v�z�}�z�n�a�U�U�U�U�U�U������������&�1�H�T�a�m�������m�T�/��������(�4�7�=�7�4�(��������ùù��������������)�2�/�3���������ù�	���������������	��-�;�G�N�H�8�/�"��	�U�M�U�]�a�n�q�o�n�a�U�U�U�U�U�U�U�U�U�U�H�F�<�/�-�/�:�<�H�K�U�a�k�e�f�d�a�U�H�H��������}������������ʼּּʼļ��������G�C�;�.�"����"�.�9�;�E�G�T�U�W�T�G�G� ���
��������
�#�/�5�<�=�>�A�B�4�/� ���ֺɺźɺֺ������ ����������;�4�������	��.�;�F�J�P�K�V�[�W�T�G�;���������������	��/�?�F�C�F�E�?�;�"�	���ϹϹ͹Ϲܹ�����ܹϹϹϹϹϹϹϹϹϹϿ`�]�T�L�T�W�`�g�m�y�������������y�m�`�`àÝÞàìù��úùìàààààààààà�T�H�L�T�W�\�g�m�y���������������y�m�`�T�����������(�4�N�������������N�5��Ňŀ�{ŇŎŔŠţŦŧŠŔŇŇŇŇŇŇŇŇ��%�A�K�L�A�(��ݽ��������������ƽԽ�ĳĬĦĥĦĳĿ��������Ŀĳĳĳĳĳĳĳĳ�ʾȾȾѾ׾޾���	�����	�����׾�ŞňňŒŠŹ������������������ŹŞ��������������������	������������<�3�#�����#�0�<�U�n�{Ł�{�s�n�b�I�<��ĳč�W�E�F�Tāĳ���������������������)�%��)�5�9�B�D�B�5�)�)�)�)�)�)�)�)�)�)E�E�E�E�F'F;FJFVFcFoF�F�F�F�FtFVF=FE�E�������'�@�L�e�r�~���������r�Y�@�3����������������������������������������������������������ɺ���� �����ֺɺ����~�x�}�������������������������������������������������������������������������x�n�_�N�J�<�D�S�_�x�������������������x�ӿѿ̿ѿԿݿ��������%����������������������������������������������������r�Y�4��'�4�Y�t�������������ȼ˼��������!�:�S�l�����о�%�!��齞�y�S������������������"�)�+�%�#�)�1�+�����p�c�T�M�K�Z�g�s�v��������������������"�	� ������.�;�a�m�p�������������a�;�"�ܻѻлƻлܻ�����������ܻܻܻܻܻ��*�����*�6�7�@�6�*�*�*�*�*�*�*�*�*�*�H�D�;�8�;�E�H�H�K�T�_�T�H�H�H�H�H�H�H�H��ƷƳƯƳ���������������������������׾;վ׾���� ������׾׾׾׾׾׾׾׾�����������	��"�+�*�#� ���	����������
���*�1�6�C�E�L�G�C�6�*�������������Ѽ�����!�%�'�$������ʼ�Ɓ�y�u�n�uƁƎƚƝƚƗƎƁƁƁƁƁƁƁƁD�D�D�D�D�D�D�D�E7ECEGE<EFEHEEE;EEED�ŠŖŔŌŔŘśŠŭŹ����������������ŭŠ��ݿڿٿܿٿ������(�*�.�+�(������0�.�#�����#�0�8�<�>�<�6�0�0�0�0�0�0���������������������������������������n�a�R�H�;�4�<�H�U�a�n�zÇàëàÞÓ�}�n < . = e @ 8 & V b N + � U E N C b L � J  T * L C � 2 @ < ' [ M g Z [ R ^ s X 3 B 6 . U h Z Y � Q l Q � Y H � ) K ^ S * 8 J P �  �  q  �  T  �  %  j  m  �  5  �  s    0    �  �    o  �  �  	  X  y  �  h  �  �  �  �  A  �  4  �    �  �  �  �  �    �  p  U  �  �    �  �  �  R  E  �  �  R  q  �  M  �  �    �  �  '<�t���1�C���`B��h��t����ͼ����D����񪼋C���1���
��9X��C���󶼬1���ͼ�j��h�ixռ�1������`B�@��'�`B��S���P���ixս'@����T�C���/������w�]/�#�
�49X��D���'�h���ٽe`B�49X��^5�H�9�D���P�`�q���q�����罕��Ƨ�aG������������ě��\��`BB
e�Bh%B��B"7B#��BB�B][B!saB!�B-BNeA�P�A���BRXBSB6�B!{lB�B��B/3mB
9Bi]B }Bn#B�A���B(AB'�A��BF�B, B��BdBhB�B��B��BP�B">�B�B��BSQB)�~B4�B�VBAMB	XaB'��B5vB�B
��B�B5(BdB	�B>B-�EB	5+B��B�B$BgAB�nB+UB
xB�B��B!�1B$C5B;9BF�B!��B �QB?�B@A���A�`�BABƋB>�B!�dBIB��B.�_B
�BH B��B��B�SA���B�B'C$A�jOB@�B>�B�0B��B��B9,B��B?�B�mB"B�eB��B?B)��B;	B��B��B	�B(|TBAUB8�B B=WB@B<B
HB��B-��B	=�B�B<sBəB��B
�BDA� A���@Agb@��AI�}A>�A�e*A*�wAƙ+A� �A7�A���A��A�v�A�"<@�4Ab)A���@EI>A`�JA�*�>��Al$�A�w�AmA�G|A�A,��A�d!AW(�A��A��FA�A�~A�E�C��i?׍�A�8p@BeA��KA��@�;A�^>A��@��2A��A�moA�BfA��@��WA���A���B�1AU�A[�A��A�zB~�C�j�A���A�t�A��A�]'AǼ�A��ZA���@C͠@��AIA=*KA�5	A)�xA�	�A���A6��A�j�A���A�|�A�yI@���Aa�)A��K@D�Ab��A��|>���AkeẢ�AmfA��A�}"A.S�A�prAV�A��A�|eA���A�xrA���C���?��A�{�@2��A���A�W'@�S�A�ZA���@��A�&A��/A��A��*@�ȶA�u�A�x�B>�AT�0A[CA�{�Ag�B��C�i%A�� A�q0A�(A�3�AǕX         #                     `   
      
            	      	      )         
            X                  4      N   -               1         Q   U         5                  $      /      L                                  '      '      ;                              !               !   3      6         %         5      3   +      !         !         5   G   #      5                  '      #      )                                       %      )                                                3      1                  /      /   %                        5   G   !      )                              #               N���Ox�iO�NAp7O�}�O�+�O���O��6Nw��P��NuRrN��OfKNggN�`N��iN�ON{�N$�OM�O�yxM�O!(LNRliO���P+�N���P7N�@�O?\bO���N�7O�P-1-M��PA�KO�GLO��O�O�N���N�_>O���O��N,V�P3`)P�O�H7O"�}PM�Nb�=NJ�LN��O,�QNX6O.9O+��O��N rO�(�O4��O��<N��N���O-�    �  7    T  A  �  �  �  �  ]  :  �  5    u  f  �  �  �  �  z  �  P  �  �  �  �  �  �  �  z  �  �  6  
    �  9  =  i  �  }  Y  	�  �  �    C  �  �  �  u    �  \  2  �  7  5    .    n<�j�o���
��o���
�o�o�ě���`B�0 żo�e`B�49X�49X�D�����
�e`B�u�u��t��ě���t����㼛�������1��9X���ě��ě�����`B�+�C����C���P�o�\)�+�C���P�\)�t��t�����w���<j��w�8Q�<j�@��L�ͽy�#�L�ͽq���T���y�#�m�h��E���Q콸Q�Ƨ�mt{����������ytjmmmm)5BNQY[\[XONB5)()-.*)#�������������������������&(%
����������������������������������������������������� ���������������������������������w����������������|vw��������������������lmz���}zqmjjllllllll���		 ��������:BHOPQOB@6::::::::::,/::<>HINUa^UH</.-,,bhjt����~thabbbbbbbb��������������������)2BO[hkptvvth[OB6)()zz���������zyxzzzzzz��*6:<==9*��Y[gt���������togc[WY������������������������� �����������������������������?BEO[ht������th[OGB?;Haz����zhYMH;31523;=BNX[b^[NB97========0<IU_n{����bUI<3.&&0*/9;<EF@;/"$********46BOX[^a[WOB660-*+.4�������������������������������������������������������������/1)������!#/31/&# !!!!!!!!!!#/HWn�����zaUH<+##���������������������������������������������������������)-54)9BOR[bc`[OJB=8999999������������������������������������~�!#,/0552/%# !!!!!!!!����
(+&)24'���������-&(>FI#�������2=FNgt�������tgXNB52IUbn{�����{nb_QKIBFIw����������������vrw)/0.+) ��������������������?BMOU[\[OJB6????????�������������������������������NO[nt|���������tgZNN������� ������������������������Y[cghkmjg\[ZYYYYYYYY�����#*7:<7#
�������������������������
#/<BEHLOJH<# 

����������������������������������������)-5CN[]WSOMHGDB950))���������������������������J�A�7�6�6�?�A�N�Z�g�l�s�������~�u�g�Z�J���������ɺֺ̺�����������ֺɺ���������r�p�r�������������������������������v�q�v�|����������ƾʾþ������������M�A�4�/�(�-�4�<�M�Z�g�s�v�~��q�f�c�Z�M�A�>�8�8�0�1�5�A�N�Z�g�s���������s�g�N�A�ݽн����������нؽ��������������U�S�H�E�H�U�a�n�v�z�}�z�n�a�U�U�U�U�U�U�"����+�2�<�H�T�a�m�y���z�l�a�T�H�/�"������(�4�7�=�7�4�(��������������������"���������������������	���������������	��-�;�G�N�H�8�/�"��	�U�M�U�]�a�n�q�o�n�a�U�U�U�U�U�U�U�U�U�U�H�F�<�/�-�/�:�<�H�K�U�a�k�e�f�d�a�U�H�H�����������������������������������������G�G�;�.�"��� �"�.�5�;�D�G�S�K�G�G�G�G�"�������
���#�/�2�;�<�=�@�A�2�/�"���ֺɺźɺֺ������ ����������G�;�)�"��	�����	��"�.�;�D�Q�T�S�I�G�����������������	�#�/�8�>�>�;�5�/��	���ϹϹ͹Ϲܹ�����ܹϹϹϹϹϹϹϹϹϹϿ`�]�T�L�T�W�`�g�m�y�������������y�m�`�`àÝÞàìù��úùìàààààààààà�m�c�`�[�X�Z�^�c�m�y�����������������y�m�����������(�4�N�������������N�5��Ňŀ�{ŇŎŔŠţŦŧŠŔŇŇŇŇŇŇŇŇ�������������нݽ���@�E�D�6��ݽ�����ĳĬĦĥĦĳĿ��������Ŀĳĳĳĳĳĳĳĳ�ʾȾȾѾ׾޾���	�����	�����׾�����ŹŭŠœŒŚŠŹ��������������������������������������������������<�9�0�%�&�0�<�I�U�_�b�n�{�u�o�n�b�U�I�<��ĳč�`�N�L�W�qāĳĿ�������������������)�%��)�5�9�B�D�B�5�)�)�)�)�)�)�)�)�)�)E�E�E�E�F*F?FJFVFcF�F�F�F�F�FsFVF=FE�E�����'�3�@�Y�r�~�����������~�Y�@�3��������������������������������������������������������ɺֺ���������ֺɺ����~�x�}�������������������������������������������������������������������������x�t�_�P�K�D�?�F�_�x�������������������x�ݿտѿͿѿֿݿ�����������������������������������������������������������r�Y�4��'�4�Y�t�������������ȼ˼��������!�:�S�l�����о�#���н��x�S�������������������$�)�#�"�$�)�/�)�����p�c�T�M�K�Z�g�s�v��������������������	������/�;�G�T�a�m�|������a�H�/�"�	�ܻѻлƻлܻ�����������ܻܻܻܻܻ��*�����*�6�7�@�6�*�*�*�*�*�*�*�*�*�*�H�D�;�8�;�E�H�H�K�T�_�T�H�H�H�H�H�H�H�H��ƷƳƯƳ���������������������������׾Ͼ׾۾������������׾׾׾׾׾׾׾׿������������� �	����"� ���	�������
���*�1�6�C�E�L�G�C�6�*���������ż׼������"���������ʼ�Ɓ�y�u�n�uƁƎƚƝƚƗƎƁƁƁƁƁƁƁƁD�D�D�D�D�D�D�D�EEE*E7ECEFECE:EEED�ŠŚŔŒŔśşŠŭŹ����������������ŭŠ��ݿڿٿܿٿ������(�*�.�+�(������0�.�#�����#�0�8�<�>�<�6�0�0�0�0�0�0���������������������������������������n�a�R�H�;�4�<�H�U�a�n�zÇàëàÞÓ�}�n < $ 9 e @ 3  T b O + K U E N 8 S L � L  T * L > � 2 H < ' ^ B [ T [ R S s Q 3 H 3 & U h Y ^ � M l Q � Y M X ) A ^ A + 8 J P �  �  �  1  T  T      )  �  �  �  �    0    �  �  �  o  �  g  	  X  y  U  h  �  a  �  �  �  �  d  u    �  :  �  N  �  �  �  @  U  �  �  �  �  �  �  R  E  �  |  �  q  T  M    �    �  �  '  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  K  s  �    q  [  A  (             �  �  �  b    �   �  �    3  7  6  +      �  �  �  U    �  �  *  �  �  �  3        	    �  �  �  �  �  �  �  �  �  �  �    &  3  @  �  #  N  A    �  �  �  �  x  b  E  *    �  �  �  <  �  K  �  
    "  .  :  @  <  3  %    �  �  �  �  �  k  =    "  �  �  �  �  �  �  �  �  �  x  R  *  �  �  �  j  5  �  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  S  F  -  �  �  i  �  �  �  �  |  f  P  :  !    �  �  �  �  �  �  �  �  �  w    c  �    [  �  �  �  �  z  =  �  �  P  �  ^  �  �  �  �  ]  Z  V  P  J  =  .      �  �  �  �  `  3    �  �  p  =  �  �  �  �  �      )  9  2  '      �  �  �  a  �  �  �  �  �  �  �  �  �    r  e  U  D  0      �  �  �  �  B    5  7  5  +    �  �  �  j  7    �  �  V    �  �  [     �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  <  F  Y  [  Y  X  g  s  l  Y  C  +    �  �  �  E  �  |  �  T  \  c  b  ]  W  R  M  J  G  F  I  K  L  N  J  E  /  �  �    �  �  �  �  u  e  S  ;  %      �  �  �  �  �  a  7  
  �  �  �  i  O  6     	  �  �  �  �  �  `  +     �  �  �  ]  �  �  �  �  �  �  �  �  �  �  �  r  ^  A     �  �  �  V    �  �  �  �  �  �  �  �  f  ;    �  �  c    �  S  �  ~  �  z  x  v  t  r  p  o  h  _  V  M  E  <  1  "      �  �  �  �  �  y  n  a  U  G  7  '    �  �  �  �  �  �  f  L  2    P  @  1      �  �  �  �  �  �  l  Q  7    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  Z  /  �  �  ;  �  �  �  �  �  �  �  �  �  t  V  @  /      �  �  �  s  R    �  �  �  �  �  �  �  �  �  �  �  �  �  q  X  1  	  �  �  G   �  �  �  �  �  �  �  �  �  �  j  =    �  �  >  �  p  �     �  �  �  �  �  �  e  F  '    �  �  �  V    �  �  `  �  �    �  �  �  �  �  �  �  �  }  g  R  >  ,    �  �  �  R     �  �  �  �  �  �  �  �  �  {  Y  (  �  �  U  �  ~  8  �  �  x  ;  j  y  t  k  Y  @  !  �  �  �  l  5     �  �  T  �  f  �  �  �  �  �  �  �  �  �  �  �  h  E     �  �  �  �  �  �  �  |  �  �  }  W  #  �  �  D    �  �  x  8  �  �  *  �  �  n  6  0  )  #          �  �  �  �  �  �  �  �  �  �  �  �  
  
  
  	�  	�  	�  	�  	�  	�  	�  	m  	  �  �  �  �  J  D  �  �  �  �  �    �  �  �  �  c  3  �  �  a    �  �  K  �  _  �  �  �  �  �  �  �  �  �  �  �  }  y  w  u  j  S  <    �  �    %  /  8  1  $     )  -  '      �  �  x  2  �  �  t  N  =  2  '      �  �  �  �  �  �  }  c  H  /    �  �  �  �  a  f  h  g  f  d  \  S  H  ;  *      �  �  �  �  �  l  J  �  �  �  �  �  {  S  *    �  �  5  �  �  2  �  -  �  �  =  {  |  q  _  L  8  #    �  �  �  �  t  L    �  �  f     �  Y  Y  Y  Y  W  N  D  ;  /  !      �  �  �  �  p  6  �  �  	�  	�  	�  	�  	m  	M  	*  	  �  �  w  F    �  �  �  /  6    �  �  �  �  �  q  a  D      �  �  s    �  C    �  �  �  4  �  �  �  �  �  �  �  �  �  �  |  e  O  7    �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  b  G  +    �  �  6  =  -    �  �  �  �  s  ;  �  �  ]    �  _  �  �  �  �  �  �  �  u  i  ]  P  B  2       �  �  a    �  e    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  [  ?  #     �   �   �   �   r  u  \  H  ;  7  2      �  �  �  �  �  v  X  6    �  �  �            	    �  �  �  �  �  m  F    �  �  �  X    �  �  �  "  @  �  �  �  �  �  �  k  9    �  n    �    [  \  W  H  :  ,  8  G  4    �  �  r  5  �  �  _  �  �    �  "  -  0  2  0  (  
  �  �  �  O    �  T  �  �    o  �  �  �  �  �  �  �  �  �  �  �  �  k  U  @  )    �  �  �  �  �  @  6  5  '    �  �  %  �  %  
�  
  	�  	  U  x  o  �  .   �  ,  2  5  4  1  ,  '      �  �  �  �  f  -  �  �  n  7  �    �  �  �  �  �  o  R  6    �  �  �  �  x  >    �    �  .         �  �  �  �  �  �  m  U  =  #    �  �  �  �  �    �  �  �  �  �  �  �  �  o  Z  E  0      �  �  �  �  p  n  T  H  2  	  �  �  �  �  �  i  >    �  �  c  n  `  ,  �