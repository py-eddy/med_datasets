CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���vȴ:     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Y�   max       P���     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <ě�     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @F
=p��     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vlQ��     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q            �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�Р         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��F   max       %        $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��t   max       B/��     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B/�     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�1�     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�)     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          C     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Y�   max       P���     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?��W���'     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       ;ě�     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��   max       @F
=p��     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @vlQ��     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P            �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?~�Q��   max       ?�ȴ9Xc     �  _�   h      /   %      
      2      :      ?      ;   	                        
      +            (      6      2   %               &   '      $      "                  	            	         	            A      '         %               6   ,   P��MN�:oO �P&��O�.7O��Nl��P?��NB��O�T�O���P���OL�PH!�NRӑN�
>P.��N�(N�A�O��/O
�NN�c<N�
OY��P�s_N7.ZO��N��
P,�_M�Y�O�(^OU�$P���O���OPN�!KOmƈO�5O�O��O/I O��OSK{O�EuN�l7N�kN�*qO:<N_�zN�!eN�H�O���O�R�N�l0O%S�N�&�N, N��N*UMO3�AP'z'ND��O�ܬO*�OZ+P-��N�}QN��DO+N�pO`�Op�^Ov�<ě�;ě�;�o�o�o��o���
�ě���`B�t��49X�D���T���T���T���T����C���C���C���t���t����
���
���
���
���
��1��1��j���ͼ��ͼ��ͼ�������������`B��h���������o�+�\)�\)��P����w��w�#�
�#�
�'',1�,1�,1�@��@��D���H�9�L�ͽL�ͽL�ͽT���Y��aG��m�h�u�y�#�y�#�����\)��������5Nimj_N������������������������������������������������������������������(.BHJSTOPB6)	
#0<ILNKI<0
	���

������������)BN[ffaZNB)���*/<HKMH</'********** "%.;HTalq{umaTH/*! ��������������������Pamz�����������zcSKP`gkmz��������zma^[[`������	������������� ������������55:@BNSVTNKB95555555)6Oajt������rbOB6/()()36BEDB;6,)((((((((u����������������vuu�����������������������
"
�������155BNU[[_e[NHB541111ggt�������tohggggggg���������������������#Ib{������nb<0��������������������������������������
"*568B=63* 
�
"IUbied[UI<0#�����������������������BN[t����������tgSLCB#(/<HVab`aUHB</)�����������������������
���������������������������	##�������`dnz����������znic``%/<HMU[aibaUH@<2/&%%��������������������'-6BO[ht}���wh[PB6*'��������������������!)6?LO[httpph\OB6&!,4HU\\UMFFCCMH</./+,mtz������������zmomm���������������������������������������� )*))#�)+56?A?65) 
�)6=@6)�
)16>B6))������������������������
#)09<<40#!
�����KUbn{��������vnf^ZUK;;?HTZ[[ZXUTHHD?<;;;ot�������������|tono��������������������
 
����

��������������������������������� ��������������������������������
��������������)6BELQUOB6���������������������vz��������������~zqv�%)($19:)����������������������35BJNSVZ[\[NJB:52033?BDMNP[gnpqh[[UNBB??��������������������)5??A=2) �IN[gt��������tg[TONI#*/9<HD@=<7/#�����s�e�Q�K�T���������������������������H�>�<�8�<�H�U�Z�a�n�p�n�m�a�\�U�H�H�H�H�����������������!�.�4�8�6�.�%�!���űŭŰŹ����������*�6�?�;�*���������a�T�H�/�"���	�����	�"�/�;�H�^�i�m�m�a���������������������ʼּۼ޼Ѽ����������z�y�v�zÃÇÓàäàÖÚÓÇ�z�z�z�z�z�z�����������w�{�����Ŀݿ��������ݿѿ�����������������������������������������������������ĽĻ���������
������
�����<�0�#�!��#�,�<�I�U�b�n�y�q�o�k�c�O�I�<��ƜƑƐƚƧ��������0�?�I�G�9� ��������)����������)�6�B�D�I�I�R�R�O�B�6�)Ç�z�t�sÃàù����������������÷ýìÇ¦²³´²­¦ŇŇ�{�n�l�e�n�{ŇőŔŘŔŇŇŇŇŇŇŇ��ݾ޾׾ʾ��������ʾ˾��.�G�_�]�Q�;��鿟���������������������������������������Y�U�S�Y�_�e�f�r�����������r�r�k�f�Y�Y�T�J�A�;�9�;�2�;�H�T�m�z��������x�m�a�TŹŹűŭūŧŭŹ��������������������ŹŹ�V�I�I�B�B�I�U�V�b�o�t�r�o�k�d�b�V�V�V�V�{�x�u�o�o�{ǈǔǗǜǛǔǈ�}�{�{�{�{�{�{�ܻԻл��ûɻлܻ�������������������{�L�6�)��N�Z���������������������������������������������T�G�=�9�?�<�G�`�m�y���{�~���������y�`�T�.�&�"��	��������	�� �"�(�.�0�;�<�;�.�Ľ������������Ľ���+�0�.�&���սнĿ����������	����������������������־˾Ѿվܾ����	��"�'�2�4�-�'��	����6�,�)�'�%�'�)�4�6�B�O�S�b�h�l�e�[�O�B�6�� �����N���������������������g�A�(�������پپ�����#�/�4�;�I�G�;�.�"��Ŀ��������������Ŀѿݿ�����ݿѿĿĺֺкɺúƺɺֺ��������ֺֺֺֺֺ��s�g�Z�S�W�Z�g�s�����������������������s�/�&�'�+�/�7�;�A�H�T�U�_�^�T�R�H�C�;�/�/���������ʺֺ����!�-�8�(����ֺɺ��_�S�J�D�D�S�i�s�x�����������������{�l�_�4�(��������(�4�A�N�R�O�M�K�A�=�4�ܹعѹù��������ùϹܹ�������������ܾZ�M�@�B�M�Z�j�s��������������������f�Z�H�"����"�1�H�M�T�W�U�O�Q�a�k�n�a�V�H�B�A�F�I�O�Z�[�h�tāĔčąā�t�h�[�V�O�B�b�Y�\�b�n�y�{ń�{�n�b�b�b�b�b�b�b�b�b�b�f�[�Z�M�G�M�U�Z�e�f�s�������s�f�f�f�f�z�w�u�y�������������������������������z�����������������������������������������������~�}�|�����������������������������x�n�l�_�_�Y�_�`�d�l�n�t�x�}�����x�x�x�x�лû����������ûлܻ���������ܻллȻ˻ǻ˻ܻ�����4�M�W�S�@�+����ݻ���������(�5�A�N�X�N�N�A�5�(� ���)�����������#�)�0�6�<�A�B�F�B�6�)�3�*�)�.�3�@�C�L�Y�]�a�Y�Y�L�L�@�3�3�3�3E7E0E7E:ECEPEXE\EPECE7E7E7E7E7E7E7E7E7E7���������!�&�-�-�7�:�A�:�-�#�!����
������������������àÓÄ�z�|�|ÃÇÓÚàããìñùúùìà�����y�m�q�x�����л��'�<�4������û�������������
���������������������4�'�%�$����$�'�4�@�M�V�^�^�S�O�M�;�4����ĿľĸĿ���������������������������̿������������Ŀѿݿ�����ݿڿѿ̿Ŀ��!�����!�:�l�������Ľܽ�۽Ľ����l�`�!E�E�E�EuEuEoEuE�E�E�E�E�E�E�E�E�E�E�E�E������"�*�6�C�J�O�U�R�O�K�C�=�6�*���������
����)�5�B�B�B�A�6�5�)�����������������������������������ĦġĚėēĒĕĚĦĳĿ������������ĿĳĦ�/�.�#�����#�/�<�H�T�U�U�S�M�H�<�7�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� : d , 9 K o D 1 Q  7 I Y D G @ s J v 8 5 > 0  [ c ^ � ; V H 4 T E > $ ) ^ 4 _ 0 M � \ g Z S v ? [ h " o k V U N E p t h m [ & W t / P  :  , <    �  �  _  �  \  �  �  ,  t  �  D  ]  �  t  q  �  �  =      .  �  �  �  5  s  
    6    A  �  M  �  )  �  �  L  	  �    1  1  �  T  3  �    z  0  �  �    6  �  -  O  %  �  �  z  �  `  p  ;  �  �    ;  2  �  �  =���
%   �0 Ž�㼬1�T���#�
�e`B�D����+�t����P��h��t���1��j�@������`B�0 Ž+�����<j�}�ě��49X��`B��%��h�����aG�������%��w�@��Y��t���C���\)�<j��C��D����O߽<j�'<j�P�`�T���D���Y���O߽ixսL�ͽ�+�ixսe`B�ixսY�������Y���9X��C���+��^5���㽛�㽓t���o��F��l����B:�B!�BV�BwB�iB%�B"�B:^B��A��tB&B �&A�@ B;�B��B�B-�B�/BWEB;�B�B�B
2>B 7B&~�B)��B+��B/��B&-B"�B	��B��B�FB54B��BD�BT$B#XB�B:MB ��B��B8B hB�B�_B�ZBC�B].B�VB!�!B$��B(�pA� B
�gBΰB�B#�B�	B7�BRB�>B+zBF`B)9B�	B�iB�B��B
�BC�B	��B6�B��B!��B@B5VBJ�B&3�B�B��B��A�i@B/FB�A�fBAaB�$B�BtB��B?�BwB��B�B
?�B�oB&��B)��B+@IB/�B&A�B=nB
hB�B�bBH�B�.B@dB<�B�B��B�MB �EB?2B?�B B9B@�B��B�.BزBC�B(B!��B$>�B)<\A���B
��B�	B9]B#��BJBO}B<�BqB�9B�jB@�B��B�/B�PB@|B
�TB>�B
5�B?�A��\AŋA�A�$NA��L@��5A�~�Av��A�%~A��OA���B��A�RA�VA�a8A��DA[$�Ar�\@�mA��ZA��kB�YB��@��A�A�88AiF�A]�uA* A��9AY��Aؙ�A���A[�zAy��@@��A�D=A��@L��@�>�A8~D>���A?�oA�8A�T�A�s]AA�gA���A�5�A��Y@�5�@���@�p�A���AղZ?�s.C���@i^�B�]A�i�@�ϘA0E�@�f�A䇒AzS"A��C�gB &�A��A�n�A��A��C�1�A��A��9A�A��QA��j@��A�a�Au�A�^�A��A�}yB	8�A�eA�wSA�|�A�^�AZ��As�@�mA�z�A�y\B��B�@���A��IA�^�Ah�"A\�A/3�A��NAZ�UA؉�A�yAZ�yAy�@<��A��$A���@K�4@��DA8�b>��A>�4A�{�A�pUA�+AAdsA���AИ:A���@���@���@�c9A�	UAՄ�?��iC��@k/fB�
A��@�*�A1�@��A�3Az�zA�jC��B >BA�3EA�pzAီA§�C�)   i      0   %      
      3      :      ?      <   	                              +            )      7      2   &               &   (      %      #         	         	            	         
            B      '         %               7   ,      7         +   #         +      !      3      1         5                        ?      #      /      #      C   !               #   %            %                           %                        1               3                                 #            !            3               !                        =            /            ;                  #                                          %                        1               3                     O� N�:oO��O�5�O] �O��NȋO��NB��O��+O���P���OL�OrX,NRӑN�
>O�!�N�(N�A�O.�O
�NN�c<N���O�P�_hN7.ZO-G�N��
P,�_M�Y�O:ܤOS�P���O���N���N�!KO5�zO�5O�/�O���O ?�O[�iN�, O�NǚyN�kNw�O:<N_�zN�!eN�H�O�7�O�R�N�l0O%S�N)yN, N�@�N*UMO3�AP'z'ND��OP�>OD�OZ+P-��N�}QN��jO+N�pO��N��Ov�  �  U  
.  j  d  �  �  	  6  	�  �  n    X  �    �  a  �    s  B  o  (  �    [  �  w  �  �  �    �  �  \  b  4  q  �  �    �  �  (  �  �  J  �  q  �    �  �  �  �  F  ;    �  	2  �  �  �  6  L  �  8  �  
  x  M  m��9X;ě��o�T���ě���o�ě���1��`B��t��49X�D���T���0 żT���T��������C���C���j��t����
��1��/��1���
��h��1��j���ͽD������h��h��/��`B�o���o�\)�C���P��P�8Q�t���P��w��w��w�#�
�#�
�@��',1�,1�D���@��D���D���H�9�L�ͽL�ͽaG��Y��Y��aG��m�h�}�y�#�y�#���㽰 Ž�����)5BHNOB5�����������������������������������������������������������������!)/6BHOPKKDB6)	
#0<ILNKI<0
	���


�������������)5BN[`a`[NB)
*/<HKMH</'***********4;HTagntumaTH9/'%'*��������������������Pamz�����������zcSKP`gkmz��������zma^[[`����������������������� ������������55:@BNSVTNKB9555555516BOht���}yh[OB751.1()36BEDB;6,)((((((((u����������������vuu�����������������������
"
�������155BNU[[_e[NHB541111kt�������tpjkkkkkkkk���������������������#Ib{������nb<0�������������������������������������������
"*568B=63* 
�
"IUbied[UI<0#�����������������������X[\got���������tgd[X#/1<DHUWWURH</-#"���������������������

���������������������������	##�������fnz����������znlebaf%/<HMU[aibaUH@<2/&%%��������������������/6BOYhy���th[SIB6,)/��������������������)6CO[hqqmmlh[OB60)$)//9<HUXYUQHD<5/.////yz������������zwstvy����������������������������������������)("	�)+56?A?65) 
�)6=@6)�
)16>B6))�������������������������
#)/32+# 
����KUbn{��������vnf^ZUK;;?HTZ[[ZXUTHHD?<;;;ot�������������|tono��������������������
 
����

	���������������������������������� ��������������������������������
������������)6BIOQOB6)��������������������vz��������������~zqv�%)($19:)����������������������55BGNQUWPNNB<5325555?BDMNP[gnpqh[[UNBB??��������������������).55:5,)agit��������tjgaaaaa#*/9<HD@=<7/#�����u�n�l�l�n�u�������������������������H�>�<�8�<�H�U�Z�a�n�p�n�m�a�\�U�H�H�H�H���������!�(�.�/�4�3�.�!�����������źŻ�������������)�*����������;�/�"���	�
��&�/�;�H�T�X�c�i�g�a�H�;���������������������ʼּۼ޼Ѽ����������z�z�w�zÆÇÊÒÓ×ÓÇ�z�z�z�z�z�z�z�z�Ŀ������������������������ѿ����ѿ�������������������������������������������������Ŀ�����������	����
����������<�0�#�!��#�,�<�I�U�b�n�y�q�o�k�c�O�I�<��ƜƑƐƚƧ��������0�?�I�G�9� ��������)����������)�6�B�D�I�I�R�R�O�B�6�)àÓÈÈÒàìù������������������ìåà¦²³´²­¦ŇŇ�{�n�l�e�n�{ŇőŔŘŔŇŇŇŇŇŇŇ�����߾Ծؾ����	�.�<�B�D�;�.�"���𿟿��������������������������������������Y�U�S�Y�_�e�f�r�����������r�r�k�f�Y�Y�T�O�H�E�@�?�H�T�a�e�m�x�z�����z�q�m�a�TŹŹűŭūŧŭŹ��������������������ŹŹ�V�I�I�B�B�I�U�V�b�o�t�r�o�k�d�b�V�V�V�V�{�w�q�q�{ǈǔǖǛǚǔǈ�{�{�{�{�{�{�{�{�ܻܻлϻϻٻܻ����
�����������������|�M�8�+�.�N�s���������������������������������������������`�T�L�G�D�B�B�G�T�`�b�m�n�y�����y�w�m�`�.�&�"��	��������	�� �"�(�.�0�;�<�;�.�Ľ������������Ľ���+�0�.�&���սнĿ����������	�����������������������������������	���"�#����	���6�0�*�)�(�)�*�4�6�B�F�O�[�g�_�[�P�O�B�6�����"�N�����������������������g�(�����۾ܾ����	��"�-�2�;�C�;�"�	����Ŀ��������������Ŀѿݿ޿��ݿѿĿĿĿĺֺкɺúƺɺֺ��������ֺֺֺֺֺ��g�]�V�Y�Z�g�p�s���������������������s�g�/�&�'�+�/�7�;�A�H�T�U�_�^�T�R�H�C�;�/�/���������˺ֺ����!�-�5�&����ֺɺ��S�L�F�F�S�l�x�������������������x�l�_�S�(� ������(�4�A�I�M�N�M�L�E�A�4�(�(�ܹչù������ùϹܹ�����������������ܾZ�Y�M�K�D�H�M�Z�c�f�q�s�~�s�j�f�Z�Z�Z�Z�"� �����"�-�;�H�J�S�T�W�X�T�H�;�/�"�O�D�G�J�O�[�\�h�tāċĂā�t�l�h�[�T�O�O�b�Y�\�b�n�y�{ń�{�n�b�b�b�b�b�b�b�b�b�b�f�_�Z�N�Z�f�s�������s�f�f�f�f�f�f�f�f�z�w�u�y�������������������������������z�����������������������������������������������~�}�|�����������������������������x�n�l�_�_�Y�_�`�d�l�n�t�x�}�����x�x�x�x�ܻлû����������ûлܻ����
��
���ܻлȻ˻ǻ˻ܻ�����4�M�W�S�@�+����ݻ���������(�5�A�N�X�N�N�A�5�(� ���)�����������#�)�0�6�<�A�B�F�B�6�)�@�8�3�/�3�9�@�L�P�R�L�@�@�@�@�@�@�@�@�@E7E0E7E:ECEPEXE\EPECE7E7E7E7E7E7E7E7E7E7���������!�)�-�5�:�?�:�-�!������
������������������àÓÄ�z�|�|ÃÇÓÚàããìñùúùìà�����y�m�q�x�����л��'�<�4������û�������������
���������������������'�'�!��!�'�4�@�M�Q�Z�Y�U�Q�N�M�J�@�4�'����ĿĹĿĿ���������������������������̿������������Ŀѿݿ�����ݿڿѿ̿Ŀ��!�����!�:�l�������Ľܽ�۽Ľ����l�`�!E�E�E�EuEuEoEuE�E�E�E�E�E�E�E�E�E�E�E�E������%�*�6�C�O�P�O�I�C�:�6�*�����������
����)�5�B�B�B�A�6�5�)�����������������������������������ĳĨĦĚĚĖėĚĦĨĳĿ������������Ŀĳ�/�&�#���#�$�/�<�H�I�L�L�H�E�<�/�/�/�/E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� 8 d + : D o < " Q  7 I Y H G @ F J v % 5 > +  W c N � ; V ) + O ; = $  ^ 4 a & F = ) c Z ^ v ? [ h / o k V 3 N 8 p t h m [   W t / H  :   <    �  �  "  �  �  �  :  �  t  H  D  ]  �  �  q  �  �  =    r  .  �  �  /  �  s  �    6    �  I    `  �  �  x  L  �  �    �  �  Z    3  p    z  0  �  �    6  �  >  O  �  �  �  z  �  �  Q  ;  �  �  �  ;  2  P    =  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  w  0  �  4  {  �  �  �  �  �  �  o  &  �  \  �  �    �  U  Q  M  H  B  ;  3  (      �  �  �  �  �  �  �  l  =    
  
  
)  
,  
"  
  	�  	�  	�  	H  	  �    s  �    f  �  �  =  �  �    @  X  f  h  \  G  %  �  �  �  W    �  M  �  .  �  M  O  T  ^  d  ^  U  K  @  3      �  �  t  /  �  �  M  
  �  �  �  �  �  �  �  �  �  �  �  �  y  d  L  5    	  �  �  a  w  �  �  �  �  �  �  �  �  �  �  �    v  m  W  =  #  	  �    �  �  �    	    �  �  �  J  �  �  �  k    �  4  �  6  0  *  $               �  �  �  �  �  �  �  �  �  �  	l  	�  	�  	�  	�  	�  	p  	G  	  �  w    �    y  �  9  r  *  '  �  �  �  �  �  x  O  %  �  �  �  n  )  �  �  S    �  �  p  n  V  8    �  �  u  ;    �  m    �  �  `  �  `  �  �  �        �  �  �  �  �  y  b  O  K  d  c  O  4  ,  
  �    �    K  c  u  �  �    /  O  W  A    �  v  �  0  v  �  %  �  �  �  �    j  N  3    �  �  �  Q    �  {  )  �  �  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  .  f  �  }      �  �  �  �  t  `  M  ;    �  �  ;  �  t  (  a  \  V  Q  L  G  B  =  8  3  -  &        
     �   �   �  �  �  �  �  �  �  �  �  6  k  c  Y  L  9  #  	  �  �  �  �  �  �              �  �  �  �  �  T  -    �  e  �  �  s  Z  D  0      �  �  �  �  �  �  �  �  e  H  3  �  b  1  B  >  8  0  )  "    �  �  �  �  Z  .  �  �  �  i  3    �  n  o  o  k  e  Y  L  ;  (    �  �  �  z  8  �  �  7   �   K  �  �  
    %  (    	  �  �  �  �  �  S    �  �  ,  �  �  �  {  J    �  |  8  �  �  �  �  g  *  �  �  e  
  �  >   �        �  �  �  �  �  �  �  �  �  t  ^  G  ,     �   �   �  *  $  '  5  H  V  [  Y  P  =  #    �  �  �  _  +  �  a   �  �  �  �  ~  |  {  n  Y  D  :  1  %    �  �  �  u  4   �   �  w  m  B  V  .    �  �  k  A    �  �  �  �  f  �  b  �   �  �  �  �  �  �  �  �  u  a  N  6    �  �  �  �  �  f  F  '  �  %  m  �  �  6  `  �  �  �  j  I    �  �  &  �  �  5  �  �  �  �  �  �  �  �  c  =    �  �  V    �    ~  �  �  �        �  �  �    X  +  �  �  �  h  >     �  b  2  �  �  �  �  �  �  �  �  �  �  �  z  W  /  �  �  �  6  �  �  ;  �  �  �  �  �  �  �  �  l  T  8    �  �  �  �  �  �  |  h  R  \  R  G  :  (    �  �  �  h  1  �  �  �  \  ,  �  �  �  �  ;  ;  O  a  P  ,    �  �  �  g  @    �  �  T  �  k  �  t  4  /  )  $          �  �  �  �  �  �  �  �  �  w  ^  E  q  o  g  T  ?  &    �  �  �  `  #  �  �  "  �  @  �      �  �  �  �  �  x  X  3    �  �  Q  �  �  7  �  o  �      �  �  �  �  �  �  l  V  =    �  �  �  y  G    �  w  *   �  �            �  �  �  F  �  �  �  �  �  X  �  i  �    o  |  u  _  Z  �  �  �  �  �  j  N  .    �  �  x  0  �  �    b  j  �  �  �  �  �  �  z  H    �  �  %  �  O  �  A  �  "  &  $      �  �  �  �  j  E    �  �  �  c  )  �  �  ^  �  �  �  �  {  w  y  z  {  }  s  ]  G  1    
  �  �  �  �  �  �  �  �  �  �  �  �  z  f  M  0    �  �  �  }  7  �  9  J  /  *  E  B  5    	  �  �  �  �  �  j  >    �    \  �  �  �  �  �  \  E  ?  E  I  M  R  c  v  �  �  �  �  �  #  N  q  d  X  N  E  >  =  <  1  $      �  �  �  �  �  �  �  �  �  �  �  �  w  e  R  =  '    �  �  �  R    �  Y  �  �  C  l  p  v  }    |  o  ]  K  A  :  ,    �  �  g  �  l  �  c  �  �  �  �  �  �  �  �  �  �  u  o  �  {  K  	  �  U   �   l  �  �  �  �  }  q  c  V  G  9  *         �  �  �  �  �  �  �  �  �  y  n  Y  8    �  �    D  	  �  �  7  �  ;  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  W  5    �  �  n  6  F  >  6  0  *  "            �  �  �  �  �  %  �  (  �  5  8  :  7  3  ,  "       �  �  �  �  �  �  `  ?    �         �  �  �  �  �  �  �  |  d  L  %  �  �  �  g  @     �  �  j  M  1    �  �  �  �  �  �  w  [  5    �  |  +  �  k  	2  	$  	  		  �  �  �  H  �  �  K  �  �  0    �  E  }  �  w  �  �  �  �  �  �  �  �  �  �  �  z  j  W  ?  '    �  �  �  e  �  �  �  �  �  x  i  N  .    �  �  V  �  �  /  �  �  �  q  �  �  t  `  K  4    �  �  �  �  �  |  g  T  F  <  #  �  6  $    �  �  �  �  �  �  p  W  8      �  �  �  Z  �  V  L  <  '    �  �  �  `     �  �  k  4  J    �  �  y  �  |  �  }  m  W  <    �  �  �  _     �  �  T    �  *  u  �   �  %  .  6  -    �  �  �  �  Q    �  �  �  X     �  �  �  �  �  �  �  �  �  �  i  K  '    �  �  �  {  [  2    �  �  �  
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
�  -  V  n  w  m  Q    
�  
z  
  	�  	W  �  Z  �  �    �   �  !  �  �  �  '  =  J  M  I  :    �  �  ?  �  k  �  w  �  �  m  a  Q  <  #  
  �  �  �  v  E    �  �  5  �  /  �  �  o